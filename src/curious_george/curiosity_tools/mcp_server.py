"""
MCP server wrapper around curiosity_tools - the service-shape boundary
described in README.md's "piper_assistant integration" roadmap section.

Everything else in curiosity_tools is library-shaped: callers pass in a
live model/tokenizer/embedder they loaded themselves. A service can't work
that way - a caller across an MCP connection (piper_assistant) can only
send and receive plain, JSON-serializable data, never a live torch model.
This module is the thin seam where that translation happens, and nothing
else: it owns exactly one warm base model + tokenizer + embedder (loaded
once, at server startup, reused across calls) for the cheap operations
that benefit from staying warm, and otherwise calls straight through to
the existing curiosity_tools functions unchanged.

Deep-scoring and real study sessions still reload a fresh base model
internally, once per trial - see deep_scoring.py and study.py's own
docstrings for why that's deliberate (clean measurement, no
cross-trial contamination), not an oversight this wrapper should paper
over. Only the cheap, single-forward-pass operations (candidate baseline
loss, embedding) actually gain anything from a shared warm model, so
that's the only part made warm here. See obsidian/Journals/2026-09-19.md
for the design discussion.

Run directly to serve over stdio:
    python -m curious_george.curiosity_tools.mcp_server

Configuration is via environment variables, matching the
CURIOUS_GEORGE_MEMORY_DIR convention memory_store.py already uses:
    CURIOUS_GEORGE_MODEL   base model for cheap readings, deep-scoring,
                            and study (default: Qwen/Qwen2.5-0.5B-Instruct)
    CURIOUS_GEORGE_DEVICE   "cpu" or "cuda" (default: "cpu")
"""

import os
from dataclasses import asdict
from typing import Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from transformers import AutoModelForCausalLM, AutoTokenizer

from curious_george.curiosity_tools import curiosity_score, deep_scoring, pipeline, research_trigger, study
from curious_george.curiosity_tools.embeddings import SmallModelEmbedder
from curious_george.curiosity_tools.memory_store import MemoryItem, MemoryStore, PipelineStatus
from curious_george.curiosity_tools.my_interests import embed_my_interests, load_my_interests

MODEL_NAME = os.environ.get("CURIOUS_GEORGE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get("CURIOUS_GEORGE_DEVICE", "cpu")

# mcp>=2.0 renamed FastMCP to MCPServer (see requirements-mcp.txt) - this
# targets that current API, not the v1 FastMCP shape older examples show.
mcp = MCPServer("curious-george")

# Populated by _startup() before mcp.run() - a plain module-level dict
# rather than a class, since there's exactly one of each of these per
# server process and every tool function needs the same handful of them.
_state: Dict[str, object] = {}


def _startup() -> None:
    """Loads everything that's expensive to load but cheap to reuse, once,
    for the life of the server process - the actual warm-service half of
    the library/service split this module exists to draw. Anything a tool
    function needs that ISN'T here (a fresh model for a deep-scoring trial
    or a study session) is loaded fresh, inside the existing function that
    already does that, exactly as it does today outside the MCP wrapper."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    embedder = SmallModelEmbedder(device=DEVICE)

    interests = load_my_interests()
    interest_embeddings = embed_my_interests(interests, embedder)

    _state.update(
        model=model,
        tokenizer=tokenizer,
        embedder=embedder,
        interest_embeddings=interest_embeddings,
        store=MemoryStore.load(),
    )


def _store() -> MemoryStore:
    return _state["store"]  # type: ignore[return-value]


def _serialize_item(item: MemoryItem, include_content: bool = True) -> dict:
    """MemoryItem -> plain dict, the shape every tool below actually
    returns. loss_history's (timestamp, float) tuples and
    DeepScoreRecord's dataclass fields both need an explicit conversion -
    neither crosses an MCP connection as-is."""
    store = _store()
    latest_deep_score = store.latest_deep_score(item.topic)
    return {
        "topic": item.topic,
        "content": item.content if include_content else item.content[:120],
        "status": item.status.value,
        "mastery": store.mastery_level(item.topic),
        "first_learned": item.first_learned,
        "last_studied": item.last_studied,
        "loss_history": [[ts, loss] for ts, loss in item.loss_history],
        "latest_deep_score": asdict(latest_deep_score) if latest_deep_score else None,
        "deep_score_count": len(item.deep_score_history),
    }


@mcp.tool()
def list_topics(status: Optional[str] = None) -> List[dict]:
    """Lists every topic in the memory store, optionally filtered to one
    PipelineStatus ("candidate", "active", "shelved", or "archived").
    Content is truncated in this list view - use get_topic for the full
    text."""
    store = _store()
    items = store.items
    if status is not None:
        items = [item for item in items if item.status.value == status]
    return [_serialize_item(item, include_content=False) for item in items]


@mcp.tool()
def get_topic(topic: str) -> dict:
    """Full detail on one topic, including its complete content and deep-
    score history. Raises (as a tool error) if the topic isn't in the
    store."""
    item = _store().get_item(topic)
    if item is None:
        raise KeyError(f"no memory item for topic {topic!r}")
    return _serialize_item(item, include_content=True)


@mcp.tool()
def add_candidate_topic(topic: str, content: str) -> dict:
    """Adds one new candidate: a cheap baseline-loss reading plus an
    embedding, no training - the entry point for anything proposed as
    worth being curious about. Lands at status="candidate"; won't compete
    for study time until run_deep_scoring promotes it to "active"."""
    result = pipeline.add_candidate(
        _store(), _state["model"], _state["tokenizer"], DEVICE, _state["embedder"],
        topic, content, interest_embeddings=_state["interest_embeddings"],
    )
    _store().save()
    return result


@mcp.tool()
def run_deep_scoring(topics: Optional[List[str]] = None, keep_top_n: Optional[int] = None,
                      num_trials: int = 5, num_steps: int = 200) -> Dict[str, dict]:
    """The expensive transferability check: num_trials LoRA trials per
    topic (each reloading its own fresh base model - see this module's
    docstring), averaged, then ranked. Defaults to every "candidate"-
    status topic never yet deep-scored; pass an explicit topics list to
    rescore specific "active" topics instead (e.g. after further study).
    keep_top_n=None (the default) keeps every measurable topic and prunes
    only the unmeasurable ones - see deep_scoring.run_deep_scoring_pass
    for the full ranking rationale. This is the slowest tool here - each
    topic costs num_trials fresh model loads plus num_steps LoRA steps
    apiece, so a multi-topic call can take minutes."""
    results = deep_scoring.run_deep_scoring_pass(
        _store(), MODEL_NAME, DEVICE, keep_top_n=keep_top_n,
        num_trials=num_trials, num_steps=num_steps, topics=topics,
    )
    _store().save()
    return {
        topic: {"outcome": outcome["outcome"], "record": asdict(outcome["record"])}
        for topic, outcome in results.items()
    }


@mcp.tool()
def rank_active_topics() -> List[dict]:
    """Ranks every "active" topic by curiosity_score (transferability +
    resonance, with a hard penalty once mastered) - the actual "what am I
    most curious about right now" ordering. Returns
    [{"topic": ..., "score": ...}, ...] highest first; topics without a
    deep score yet (score is None) are left out, matching
    curiosity_score.rank_candidates."""
    ranked = curiosity_score.rank_candidates(
        _store(), interest_embeddings=_state["interest_embeddings"],
        embedding_method=_state["embedder"].METHOD_NAME,
    )
    return [{"topic": topic, "score": score} for topic, score in ranked]


@mcp.tool()
def study_topic(topic: str, num_steps: int = 200, learning_rate: float = 1e-4) -> dict:
    """Runs one real, committed LoRA study session on an "active" topic,
    persisting the result into its loss_history (see study.study_topic).
    Raises if the topic has never been deep-scored ("candidate") or is
    already archived - both are refused deliberately, not silently
    no-op'd."""
    result = study.study_topic(_store(), MODEL_NAME, DEVICE, topic,
                                num_steps=num_steps, learning_rate=learning_rate)
    _store().save()
    return result


@mcp.tool()
def run_research_trigger(topic: Optional[str] = None, num_steps: int = 200,
                          short_name: Optional[str] = None) -> dict:
    """The RESEARCH trigger (see research_trigger.py): the one tool meant
    to be called during Piper's RESEARCH mode. With topic=None (the
    default), it proposes new candidates from suggestions.json, deep-
    scores anything not yet deep-scored, enforces the active-tier capacity
    (ACTIVE_CAPACITY=5, shelving whatever's displaced), picks what to
    study via the breadth-then-depth policy, studies it for num_steps, and
    logs the run to data/research_log/. Pass topic to instead study that
    exact "active" topic directly, skipping all of the above - a human-
    directed override, not the default path. This is also the slowest
    tool here when topic=None and there are new/unscored candidates -
    deep-scoring runs inline before selection."""
    result = research_trigger.run_research_trigger(
        _store(), _state["model"], _state["tokenizer"], MODEL_NAME, DEVICE, _state["embedder"],
        interest_embeddings=_state["interest_embeddings"],
        topic=topic, num_steps=num_steps, short_name=short_name,
    )
    _store().save()
    return result


if __name__ == "__main__":
    _startup()
    mcp.run()
