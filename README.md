# curious-george

A research project attempting to give an existing LLM something that behaves
like human curiosity — not as a metaphor, but as a measurable mechanism: a
policy for choosing what to study next, and a way to verify afterward whether
studying it actually taught the model anything.

Runs as a Google Colab notebook project (Colab Pro, NVIDIA GPU) rather than
on constrained edge hardware, both because the fine-tuning experiments here
need real compute and because a notebook is a good learning format for
working through the ideas step by step.

## The core claim, made precise

Most things that get called "curiosity" in AI systems are actually novelty-
seeking or entropy-maximization: try whatever is unfamiliar, or whatever the
sampling temperature happens to surface. That's not curiosity, it's noise
with a good PR department — an earlier pass at this exact idea (see the
project history in `obsidian/Journals/` on the `piper_assistant` repo)
collapsed into exactly that failure mode, and this project starts over
specifically to avoid it.

The starting proposition — "curiosity targets what's vaguely understood, not
what's unknown" — turned out to need a real correction, not just a sharper
definition. **Baseline loss alone does not predict whether studying
something is worthwhile.** A real, 5-trial-replicated experiment (see
Phase 1 below) found that real-world facts (low loss) and pure noise (high
loss) both show *negative* transfer to related material when studied — only
content with genuine template structure showed positive transfer, regardless
of where its baseline loss sat. The actual dividing line is **transferability**
— does studying this generalize to related material, or does it only ever
memorize exactly what it saw (or worse, actively corrupt nearby predictions).
That's the concept this project now treats as "curiosity-worthy," validated
against real models rather than assumed:

| Concept | What it measures | Why it's not enough alone |
|---|---|---|
| **Novelty** | How unfamiliar the content is | Chases noise, incoherent text, garbage |
| **Uncertainty / baseline loss** | How poorly the model currently predicts it | Proven insufficient — `known` (low loss) and `noise` (high loss) both showed negative transfer in the real Phase 1 result |
| **Transferability** | Does studying this improve prediction on *unseen*, related material | The signal that actually separated learnable structure from noise in real experiments |

This is the same idea behind intrinsic-motivation work in developmental
robotics (Oudeyer et al.'s learning-progress drive) and compression-progress
theories of curiosity (Schmidhuber) — a known, studied concept — but tested
here against an LLM's own LoRA fine-tuning dynamics, with real, sometimes
surprising results (see "What didn't work" in the notebook for two real
measurement-design failures that shaped the final approach).

## What's been built and validated

Each phase below is real: run against actual models, with real numbers
recorded in `notebooks/curious_george.ipynb` and design decisions recorded
in `obsidian/Journals/`, not just planned.

**Phase 0 — Foundation.** `MemoryStore` (in-RAM, disk-persisted, brute-force
cosine similarity), a fabricated ground-truth pair (`warbles`/`quaddles`)
that closes the pretraining-leakage hole real facts can't, teacher-forced
loss measurement, and LoRA fine-tuning as the genuine "study" mechanism — a
**cold** post-study measurement (zero study content in the prompt) is the
only way to know improvement came from a real weight update and not
in-context priming. Real result: LoRA on warble facts improved recall loss
by ~2.65-2.75 nats cold, reproduced on GPU.

**Phase 1 — Does baseline loss predict learning value?** Two real
measurement-design failures (same-topic held-out facts colliding with
trained material) led to the design that actually works: pair each topic
with a separate, structurally-similar **sibling** topic, never trained on,
to measure genuine generalization instead of memorization. 5-trial-replicated
real result across `known`/`moderate`/`noise` anchor topics: only `moderate`
(genuine template structure) showed positive transfer; `known` and `noise`
both showed negative transfer, severity scaling with how little structure
was actually there. This is the result that revised the core claim above.

**Phase 2 — The interest pipeline (for real, non-fixture candidates).**
Phase 1's fixtures were hand-built, deliberately matched pairs. Real
candidate topics don't arrive that way. Built: a two-tier cost model (cheap
entry-scoring — one baseline loss reading plus an embedding, no training —
vs. expensive deep-scoring — a real 5-trial-averaged LoRA check); a
`my_interests` resonance system (5 human-declared interests, plus a
bounded, decaying `InterestPool` for anything that earns its way in via
real study, capacity 10 total); and a resolution to the "sibling problem"
for arbitrary real content — three fixed, offbeat "canary" topics
(`canary_topics.py`) stand in for a domain-matched sibling, explicitly
documented as measuring something related but different from Phase 1's
original signal (generalization-quality, not structural transfer). Deep
scoring ranks candidates by relative rank (`keep_top_n`), not an absolute
threshold — a real batch of non-fixture candidates showed an absolute
cutoff would have pruned everything, since canary-based scoring on
real, unmatched content tends to run mildly negative regardless of quality.

**Phase 3 — The combined curiosity score.** `curiosity_score` finally
combines mastery, resonance, and transferability into one ranking function,
with transferability as the dominant term (per Phase 1's own finding),
mastery only as a hard stop once something is fully learned, and resonance
as a smaller preference boost on top of genuine substance, not a
replacement for it.

**Real study (`study.py`).** Distinct from deep-scoring's disposable
measurement trials — a real, committed LoRA study session on a topic that
survived deep-scoring (`ACTIVE` status only), with the result persisted into
`MemoryStore.loss_history` rather than thrown away. Documented limitation,
stated plainly rather than glossed over: like every other trial in this
project, each session reloads a fresh base model and discards the trained
adapter afterward — nothing persists between calls, so repeated study
sessions measure "would committed study help, right now" repeatably, not
mastery actually accumulating over calendar time. True accumulation needs
an open, deliberately deferred question resolved: whether real study should
ever persist into a shared, evolving model.

**The RESEARCH trigger (`research_trigger.py`) — the loop, closed.** One
function, `run_research_trigger`, that actually runs the cycle above
end to end: ingest new candidates from `suggestions.json` (human-curated,
see `suggestions.py`) → cheap-score them → deep-score every `CANDIDATE` not
yet scored → enforce the active tier's bounded capacity (`ACTIVE_CAPACITY`
= 5, mirroring the "~5 attention threads" framing in
`obsidian/Journals/2026-09-13.md` — this bounded `ACTIVE` set *is* "the
interest pipeline," `MemoryStore` as a whole is the unbounded memory store)
via `curiosity_score`, shelving whatever's displaced rather than deleting
it (`PipelineStatus.SHELVED`) → pick what to study via a breadth-then-depth
policy (explore every `newbie` topic, strict rotation, before favoring
`curiosity_score`-ranked depth) → `study_topic` for a caller-specified
`num_steps` → log the run to `data/research_log/`. A caller can also pass
an explicit `topic`, a human-directed override that skips straight to
studying that (already-`ACTIVE`) topic. See
`obsidian/Journals/2026-09-19.md` for the four design questions this
resolved (capacity tier, the "what to research"/"intended duration"
parameters, and the selection policy) and why each landed where it did.

None of this *is* autonomous curiosity yet — a human still decides what
goes in `suggestions.json` and when the trigger fires — but the mechanism
itself now runs as one real cycle instead of isolated, manually-sequenced
calls.

## Roadmap

**Now — `piper_assistant` becomes the MCP client.** The loop is closed and
the MCP server exposes it (`run_research_trigger`, alongside the
primitives), so this is no longer blocked on anything in this repo. This
was deliberately pulled forward to run in parallel with the loop-closing
work above rather than strictly after it — waiting for a fully closed
autonomous loop before starting any `piper_assistant` work risked spending
too much time fine-tuning this repo before the integration that actually
matters gets started.

`piper_assistant`
(Jetson) adds an MCP client and a supervisory state machine (`LISTENING` vs.
`RESEARCH` mode) that calls the server above during idle time. This is
`piper_assistant`'s own repo and its own branch — not a new repo — since
it's extending Piper's existing supervisory loop, not replacing it; see
`obsidian/Journals/2026-09-19.md` for that decision. Two real architectural
differences from how this repo ran before the MCP server existed, identified
before starting rather than discovered partway through, and both already
reflected in `mcp_server.py`: (1) the existing `curiosity_tools` functions
take a live model/tokenizer/embedder as parameters — a library shape, not a
service shape; `mcp_server.py` is the thin wrapper that owns the model
internally and exposes tools taking only plain, JSON-serializable data.
(2) Every deep-scoring/study trial still reloads the base model fresh,
deliberately, for clean measurement — the wrong pattern for a responsive
service in general, but `mcp_server.py` only keeps warm the one thing that
actually benefits from it (the cheap, single-forward-pass model+tokenizer+
embedder used for candidate baseline readings); deep-scoring and study calls
still reload fresh internally, unchanged, exactly as documented in those
modules. Not refactoring that part for now — a shared, persistent adapter
across calls is future work, tracked as its own open question (see "Real
study" above).

**Later — shrink the structural-leakage confound.** Phase 0's quaddle leak
(~26% of warbles' improvement showed up on the unrelated control) is real
and unresolved. Candidate directions: more topics with more varied sentence
structure, rank/target-module ablations on the LoRA config.

**Later — scale up.** Bigger base model, more topics per session, once
Phases above are trustworthy enough to be worth the extra compute cost.

**Later (stretch) — an actual autonomous session.** Multiple
select→study→measure cycles run unattended, with a logged trajectory of
what got chosen, why, and what it yielded — the real test of whether this
produces a coherent "line of inquiry" rather than a scattershot of
disconnected facts.

## Explicit non-goals (for now)

- Autonomous candidate generation — every candidate has been human-proposed
  so far; an autonomous "notice something worth being curious about"
  mechanism is unbuilt and unscoped.
- Any actual code changes in `piper_assistant` itself — that's tracked in
  that repo, on its own branch, once this repo's MCP server surface is
  stable enough to build a client against.
- Real study sessions persisting into a shared, evolving model — every LoRA
  adapter is currently disposable by design; resolving this is a
  prerequisite for genuine cross-session mastery accumulation, not solved
  here.
- Multi-topic simultaneous LoRA training — one adapter per study step, kept
  deliberately simple.
- A single RESEARCH trigger call studying more than one topic — one topic
  per call, deliberately, to avoid an unneeded scheduling problem (which
  topic first, with what leftover budget) before there's a real need for it.
- Any richer measure of "depth of understanding" than the existing
  loss-threshold `mastery` (distinct-token breadth, cross-source overlap/
  "solidity," etc.) — a real candidate hypothesis exists (see
  `obsidian/Journals/2026-09-19.md`) but needs `MemoryItem.content` to
  become multiple sourced passages instead of one string, which needs real
  search/retrieval to be worth building — the same gap as the
  mastery-of-broad-topics deferral above, not a new one.

## Repo layout

Two subpackages under `src/curious_george/`, split deliberately so it's
always clear what's needed for the actual running mechanism versus what
only ever existed to validate that the mechanism works — see
`obsidian/Journals/2026-09-16.md`. Only `curiosity_tools` would ever need
to become an MCP tool surface.

```
src/curious_george/
  curiosity_tools/            the actual mechanism - production code
    memory_store.py             in-RAM, disk-persisted knowledge + loss/deep-score history
    embeddings.py                SmallModelEmbedder (MiniLM) + QwenHiddenStateEmbedder
    extractor.py                 ResidualExtractor - hidden-state hook utility
    loss_measurement.py          teacher-forced loss on probes + measure_content_loss (no probes needed)
    lora_finetune.py             the LoRA "study" mechanism
    my_interests.py / .json      5 protected, human-declared resonance anchors
    interest_pool.py             bounded (5 slots), decaying pool for anything promoted beyond the 5
    pipeline.py                  cheap entry-scoring (mastery + resonance, no training)
    canary_topics.py / .json     fixed offbeat siblings standing in for a domain-matched partner
    deep_scoring.py              expensive transferability check + top-N prune/promote pass
    curiosity_score.py           combines mastery + resonance + transferability into one ranking
    study.py                     a real, committed LoRA study session (not a disposable trial)
    suggestions.py / .json       human-curated candidate topics - the RESEARCH trigger's input queue
    research_trigger.py          closes the loop: propose -> score -> select -> study -> log, one call
    mcp_server.py                MCP server wrapper - the service-shape boundary for piper_assistant
  validation_tools/           proved the mechanism works - not needed once trusted
    fictional_entities.py / .json  warbles/quaddles - Phase 0's fabricated ground truth
    curiosity_topics.py / .json    known/moderate/noise - Phase 1's sibling-pair topic bank
    curiosity.py                   Phase 1's experiment harness (run_topic_trial, run_repeated_trials)
    warble_harness.py              Phase 0's end-to-end sanity check + the real LoRA experiment
tests/                       pytest suite, one file per module above
notebooks/curious_george.ipynb   the narrated, real-result record of every phase
obsidian/Journals/            dated design-decision records, including what didn't work
```

## Persisting memory across Colab sessions

Colab's local disk is wiped on every runtime restart, but the whole premise
of `MemoryStore` is that `loss_history` survives across sessions — without
that, `learning_progress` never has more than one measurement to compare
against. Point it at a mounted Drive folder instead of the ephemeral local
disk by setting `CURIOUS_GEORGE_MEMORY_DIR` before calling `save()`/`load()`
with no explicit path:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["CURIOUS_GEORGE_MEMORY_DIR"] = "/content/drive/MyDrive/curious-george-memory"

from curious_george.curiosity_tools.memory_store import MemoryStore
store = MemoryStore.load()   # picks up last session's state from Drive, or starts empty on first run
# ... study something, record_loss, etc ...
store.save()                 # writes back to the same Drive folder
```

The env var is read at call time, not at import time, so it can be set any
time before the `save()`/`load()` call — it doesn't have to happen in the
same cell as the import. Locally (tests, non-Colab use), leaving it unset
falls back to the repo-relative `data/memory/` default.

## Running the tests

```bash
pip install -r requirements.txt
pytest -m "not slow"   # fast: memory store, content/schema, loss math, embeddings, orchestration
pytest -m slow         # slow: real LoRA fine-tuning and deep-scoring runs against Qwen2.5-0.5B
```

## Running the MCP server

```bash
pip install -r requirements.txt -r requirements-mcp.txt
python -m curious_george.curiosity_tools.mcp_server
```

Serves over stdio, so it's meant to be launched by an MCP client (an
`mcp.json`-style config in `piper_assistant`, or `mcp dev` for manual
poking) rather than run standalone. Loads one base model + tokenizer +
embedder at startup and keeps them warm for the process lifetime — expect a
several-second startup delay before it's ready to accept tool calls, same
as loading the model in a notebook cell. Configuration is via the same
environment-variable convention `CURIOUS_GEORGE_MEMORY_DIR` already uses:

- `CURIOUS_GEORGE_MODEL` — base model for cheap readings, deep-scoring, and
  study (default `Qwen/Qwen2.5-0.5B-Instruct`)
- `CURIOUS_GEORGE_DEVICE` — `cpu` or `cuda` (default `cpu`)
- `CURIOUS_GEORGE_MEMORY_DIR` — where the store persists (default
  `data/memory/`; point this at the same location every restart, or the
  server starts with an empty store)

Tools exposed: `list_topics`, `get_topic`, `add_candidate_topic`,
`run_deep_scoring`, `rank_active_topics`, `study_topic` — the primitives,
unchanged in behavior from calling them directly in Python — plus
`run_research_trigger`, the one call meant to be used day to day: it runs
the whole propose → score → select → study → log cycle itself (or, given
an explicit `topic`, studies exactly that one instead). See
`research_trigger.py` and the "What's been built" section above for what
it actually does.
