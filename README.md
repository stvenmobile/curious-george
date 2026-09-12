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
what's unknown" — is the right intuition but needs a sharper definition to be
buildable. Three things get confused under the word "curiosity," and this
project treats them as three different, measurable quantities:

| Concept | What it measures | Behavior if maximized alone |
|---|---|---|
| **Novelty** | How unfamiliar the content is | Chases noise, incoherent text, garbage |
| **Uncertainty** | How poorly the model currently predicts it | Chases *unlearnable* noise just as happily as real structure — high loss doesn't distinguish "almost graspable" from "gibberish" |
| **Learning progress** | How much a study step actually *reduced* loss | Targets exactly the "vaguely understood" zone: something the model has partial purchase on, where a study step yields real, measurable improvement |

The working definition for this project: **curiosity is a preference for
topics with high *predicted* learning progress, not high uncertainty and not
high novelty.** This is the same idea behind intrinsic-motivation work in
developmental robotics (Oudeyer et al.'s learning-progress drive) and
compression-progress theories of curiosity (Schmidhuber) — it is a known,
studied concept, not a new invention — but it hasn't yet been built and
tested against an LLM's own fine-tuning dynamics in this project, which is
the actual research contribution here.

Concretely: a topic is "vaguely understood" when the model's current loss on
probes about it is *moderate* — not near-zero (already knows it, nothing to
gain) and not enormous with no internal structure to hook into (pure noise —
loss stays high even after a real study step). The signal that separates
those two moderate-loss cases is exactly `learning_progress` as already
built: measure loss before, study, measure loss after (cold, no context —
see below), take the delta. A topic worth being "curious" about is one where
a cheap trial study step produces a real delta, not just one that starts out
uncertain.

## What's already been validated (the foundation this builds on)

Ported from `piper_assistant`'s `feature/piper-memory` branch, where it was
built and tested end-to-end against a real model
(Qwen2.5-0.5B-Instruct) before this repo existed:

- **`memory_store.py`** — an in-RAM, disk-persisted store of what's actually
  been studied. Deliberately brute-force cosine similarity, not FAISS — at
  this scale (hundreds to low-thousands of topics) an approximate index buys
  nothing and costs inspectability. `learning_progress(topic)` is the core
  signal: `None` until two measurements exist, otherwise `first_loss -
  latest_loss`.
- **`fictional_entities.py` / `.json`** — a synthetic ground-truth pair
  ("warbles" studied, "quaddles" a structurally-parallel negative control)
  that closes the one hole real-world facts can't: pretraining leakage. If a
  model improves on fabricated facts it was never trained on before this
  project existed, that improvement is unambiguously attributable.
- **`loss_measurement.py`** — teacher-forced cross-entropy on a held-out
  recall probe. This is the actual "did it learn" measurement everything
  else depends on.
- **`lora_finetune.py`** — the real "study" mechanism. A LoRA adapter
  (`peft`, frozen base model, trainable low-rank matrices on `q_proj`/
  `v_proj`) is the only thing that changes. This matters for curiosity
  specifically: a **cold** post-study measurement (zero study content in the
  prompt) is the only way to know improvement came from a genuine weight
  update and not in-context priming — an in-context version of this same
  experiment showed 40%+ of the apparent "learning" was just reading-
  comprehension confound from having *any* coherent text present, not from
  the specific facts.
- **`warble_harness.py`** — ties it together. The real, already-run result:
  LoRA on warble facts improved warble recall loss by 2.65 nats cold, with
  the structurally-parallel quaddle control improving by only 0.68 nats
  (26% leak, down from ~44% when studied in-context) — genuine, mostly
  fact-specific, persistent learning, with a real and honestly-reported
  residual confound (shared sentence structure between the two fabricated
  species).

None of this *is* curiosity yet. It's the instrument curiosity needs: a way
to measure, cleanly, whether studying something specific actually helped.
Curiosity is the policy that decides what to point the instrument at next.

## Roadmap

**Phase 0 — Foundation (done, ported from `piper_assistant`)**
Memory store, synthetic ground truth, cold loss measurement, LoRA study
mechanism, validated end-to-end against a real model.

**Phase 1 — Define and implement a curiosity score**
Turn "vaguely understood" into a number. Starting proposal:
`curiosity_score(topic) = predicted_learning_progress(topic) - already_known_penalty(topic)`,
where the current recall-probe loss stands in for "already known" (low loss
= low score, nothing to gain) and *predicted* learning progress needs its
own model — initially a cheap proxy (a handful of gradient steps on a small
LoRA rank, just enough to see if loss moves at all) before investing a full
study run. This phase is where the project either confirms or falsifies the
core hypothesis: does a moderate-loss topic reliably yield more learning
progress than a low- or extreme-loss one?

**Phase 2 — Candidate generation**
A pool of topics to be curious *about* has to come from somewhere. Likely a
RAG-style ingestion pipeline (documents → candidate topic chunks → baseline
loss probe on each) rather than hand-written fictional entities, once the
scoring mechanism from Phase 1 is trustworthy. Fabricated entities stay in
the test suite as the ground-truth check Phase 1 is validated against, not
as the production content source.

**Phase 3 — The closed loop**
Select highest-curiosity-score candidate → LoRA study step → cold
re-measurement → record `learning_progress` in `MemoryStore` → let that
history influence future scoring (a topic that reliably yields progress
should raise the priority of related topics; one that plateaus should stop
attracting attention). This is the actual "behaving to reduce uncertainty
in ways that have historical structure" goal from the original design
discussion — the loop is where "learning over time" becomes real rather
than a single before/after snapshot.

**Phase 4 — Shrink the structural-leakage confound**
The 26% quaddle leak in the Phase 0 result is real and unresolved. Candidate
directions: more topics with more varied sentence structure (so shared
syntax stops being a viable shortcut), rank/target-module ablations on the
LoRA config, or a control that varies structure deliberately to measure the
leak's actual source rather than just its size.

**Phase 5 — Scale up on Colab GPU**
Move past Qwen2.5-0.5B/CPU toy runs: bigger base model, more topics per
session, faster iteration on Phases 1–4 now that GPU compute removes the
multi-minute-per-run cost that made rapid iteration painful on CPU.

**Phase 6 (stretch) — An actual autonomous session**
Multiple select→study→measure cycles run unattended, with a logged
trajectory of what got chosen, why (its curiosity score at selection time),
and what it actually yielded — the first real test of whether this produces
anything like a coherent "line of inquiry" rather than a scattershot of
disconnected facts.

## Explicit non-goals (for now)

- Real-world knowledge ingestion before Phase 2 — fabricated ground truth
  stays the validation tool until the scoring mechanism itself is trusted.
- Deployment back onto the Jetson Orin NX / `piper_assistant` — this track
  was split out specifically so it could run on Colab GPU compute without
  dragging `torch`/`transformers`/`peft` into the Jetson assistant's
  dependency footprint. Revisiting that integration is a question for well
  after Phase 3, not now.
- Multi-topic simultaneous LoRA training — one adapter per study step, kept
  deliberately simple until the single-topic loop is well understood.

## Repo layout

```
src/curious_george/
  memory_store.py         in-RAM, disk-persisted knowledge + loss history
  fictional_entities.py    synthetic ground-truth content (warbles/quaddles)
  fictional_entities.json
  embeddings.py            SmallModelEmbedder (MiniLM) + QwenHiddenStateEmbedder
  extractor.py             ResidualExtractor — hidden-state hook utility
  loss_measurement.py      teacher-forced cross-entropy on recall probes
  lora_finetune.py         the LoRA "study" mechanism
  warble_harness.py        end-to-end sanity check + the real LoRA experiment
tests/                     pytest suite ported from the original validation scripts
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

from curious_george.memory_store import MemoryStore
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
pytest -m "not slow"   # fast: memory store, fictional entities, loss math, embeddings
pytest -m slow         # slow: real LoRA fine-tuning runs against Qwen2.5-0.5B
```
