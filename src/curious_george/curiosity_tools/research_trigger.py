"""
The RESEARCH trigger - closes the loop described in README.md's roadmap:
propose candidates -> cheap-score -> deep-score -> enforce active-tier
capacity -> pick what to study -> study_topic -> log the run.

Named after the LISTENING/RESEARCH state-machine split sketched for
piper_assistant, not "idle trigger" - see obsidian/Journals/2026-09-19.md
for that naming decision and the design questions this module resolves
(capacity tier, the "what to research"/"intended duration" parameters,
and the breadth-then-depth selection policy).

Two ways to call run_research_trigger:
- topic=None (the default): the trigger proposes new candidates from
  suggestions.json, deep-scores anything not yet deep-scored, enforces
  ACTIVE_CAPACITY, then picks what to study itself via select_next_topic.
- topic="some_topic" (a human-directed override): skips all of that and
  studies exactly that topic, which must already be ACTIVE - the same
  restriction study_topic itself enforces.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from curious_george.curiosity_tools import curiosity_score, deep_scoring, pipeline, study, suggestions
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "research_log"

# The interest pipeline's bounded, "focus of attention" capacity - see
# obsidian/Journals/2026-09-13.md's original framing (~5 attention
# threads) and 2026-09-19.md's confirmation that this IS that pipeline:
# MemoryStore's unbounded history is the memory store, PipelineStatus.ACTIVE
# is the bounded interest pipeline. Matches InterestPool.OPEN_SLOT_CAPACITY
# for the same reason, though the two pools track different things (topics
# under active study here, resonance anchors there) and aren't linked.
ACTIVE_CAPACITY = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_short_name(name: str) -> str:
    """Lowercase, non-alphanumerics collapsed to underscores, truncated
    to 12 chars - safe for a filename prefix regardless of what a caller
    passes (a topic name, which may contain spaces or punctuation)."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return cleaned[:12] or "topic"


def select_next_topic(store: MemoryStore) -> Optional[str]:
    """The breadth-then-depth selection policy, over ACTIVE topics only.
    Breadth phase: if any active topic is still "newbie", pick the one
    studied longest ago (strict rotation) - not resonance-weighted, since
    weighting by resonance here would just reintroduce the rich-get-richer
    problem one level down, among slow-to-graduate favorites. Depth phase
    (no newbies left): pick the top of rank_candidates - the highest
    curiosity_score. Returns None if there are no ACTIVE topics at all
    (nothing to study yet - promote_candidates needs to run first).

    Pure function of the store's current state - no live model needed,
    so this is fully unit-testable without any real inference."""
    active_items = [item for item in store.items if item.status == PipelineStatus.ACTIVE]
    if not active_items:
        return None

    newbies = [item for item in active_items if store.mastery_level(item.topic) == "newbie"]
    if newbies:
        return min(newbies, key=lambda item: item.last_studied).topic

    ranked = curiosity_score.rank_candidates(store)
    return ranked[0][0] if ranked else None


def _enforce_active_capacity(store: MemoryStore, interest_embeddings: Optional[Dict[str, object]] = None,
                              embedding_method: Optional[str] = None,
                              capacity: int = ACTIVE_CAPACITY) -> List[str]:
    """Ranks every ACTIVE topic by curiosity_score and shelves whatever
    falls beyond `capacity` - the standing version of keep_top_n described
    in obsidian/Journals/2026-09-19.md. An ACTIVE topic without a deep
    score yet (shouldn't normally happen - promotion to ACTIVE only
    happens via deep-scoring) is excluded from ranking and so can't be
    shelved by this pass; that's an acceptable, documented edge case, not
    a silent bug. Returns the list of topics shelved this call."""
    ranked = curiosity_score.rank_candidates(
        store, interest_embeddings=interest_embeddings, embedding_method=embedding_method,
    )
    shelved = []
    for topic, _score in ranked[capacity:]:
        store.set_status(topic, PipelineStatus.SHELVED)
        shelved.append(topic)
    return shelved


def promote_candidates(store: MemoryStore, model, tokenizer, model_name: str, device: str, embedder,
                        interest_embeddings: Optional[Dict[str, object]] = None,
                        num_trials: int = 5, deep_scoring_steps: int = 200, learning_rate: float = 1e-4,
                        suggestions_path: Path = suggestions.DEFAULT_PATH) -> dict:
    """The non-study half of one RESEARCH trigger cycle: ingest new
    suggestions as cheap-scored candidates (needs the live model/tokenizer/
    embedder, for the cheap baseline-loss reading and embedding), deep-
    score every CANDIDATE not yet deep-scored (needs model_name - deep-
    scoring reloads a fresh model per trial internally, see
    deep_scoring.py), then enforce ACTIVE_CAPACITY across the WHOLE active
    set, not just this batch, via curiosity_score. Separated from
    run_research_trigger so a topic override can skip straight past all
    of this."""
    all_suggestions = suggestions.load_suggestions(suggestions_path)
    fresh = suggestions.new_suggestions(store, all_suggestions)
    added = pipeline.add_candidates(store, model, tokenizer, device, embedder, fresh,
                                     interest_embeddings=interest_embeddings)

    candidate_topics = [item.topic for item in store.items if item.status == PipelineStatus.CANDIDATE]
    deep_scored: Dict[str, str] = {}
    if candidate_topics:
        results = deep_scoring.run_deep_scoring_pass(
            store, model_name, device, keep_top_n=None, num_trials=num_trials,
            num_steps=deep_scoring_steps, learning_rate=learning_rate, topics=candidate_topics,
        )
        deep_scored = {topic: outcome["outcome"] for topic, outcome in results.items()}

    shelved = _enforce_active_capacity(
        store, interest_embeddings=interest_embeddings, embedding_method=embedder.METHOD_NAME,
    )

    return {
        "suggestions_considered": len(all_suggestions),
        "new_candidates": added,
        "deep_scored": deep_scored,
        "shelved": shelved,
    }


def run_research_trigger(store: MemoryStore, model, tokenizer, model_name: str, device: str, embedder,
                          interest_embeddings: Optional[Dict[str, object]] = None,
                          topic: Optional[str] = None, num_steps: int = 200, learning_rate: float = 1e-4,
                          short_name: Optional[str] = None, log_dir: Path = DEFAULT_LOG_DIR) -> dict:
    """Runs one RESEARCH trigger cycle and returns a result dict, also
    written to log_dir/<short_name>_<timestamp>.json. See this module's
    docstring for the topic=None vs topic=override behavior. num_steps is
    the trigger's "intended duration" parameter - passed straight through
    to study_topic's own num_steps (see obsidian/Journals/2026-09-19.md
    for why: it's the one unit the mechanism already measures against)."""
    promotion = None
    if topic is None:
        promotion = promote_candidates(
            store, model, tokenizer, model_name, device, embedder,
            interest_embeddings=interest_embeddings,
        )
        selected_topic = select_next_topic(store)
        if selected_topic is None:
            return {"outcome": "nothing_to_study", "topic": None, "promotion": promotion}
    else:
        selected_topic = topic

    score_at_selection = curiosity_score.score_topic(
        store, selected_topic, interest_embeddings=interest_embeddings,
        embedding_method=embedder.METHOD_NAME if interest_embeddings else None,
    )
    result = study.study_topic(store, model_name, device, selected_topic,
                                num_steps=num_steps, learning_rate=learning_rate)

    name = _sanitize_short_name(short_name or selected_topic)
    log_entry = {
        "topic": selected_topic,
        "short_name": name,
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "loss_before": result["loss_before"],
        "loss_after": result["loss_after"],
        "mastery": result["mastery"],
        "curiosity_score_at_selection": score_at_selection,
        "timestamp": _now_iso(),
    }

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{log_entry['timestamp'].replace(':', '-')}.json"
    log_path.write_text(json.dumps(log_entry, indent=2), encoding="utf-8")

    return {**log_entry, "outcome": "studied", "log_path": str(log_path), "promotion": promotion}
