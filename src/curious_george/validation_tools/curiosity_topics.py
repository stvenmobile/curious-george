"""
Phase 1: the memorization-vs-generalization topic bank.

Phase 0's recall_probes were always decompositions of the exact
sentences trained on, so it never actually distinguished memorization
from generalization. The first version of this module tried to test
generalization by withholding some of a topic's OWN facts from
training (held_out_facts) - a real experiment against the real model
showed that design is broken: withholding some facts about "warbles"
while training on the rest doesn't test whether the model learned a
reusable pattern, it tests whether a small-capacity LoRA adapter's
near-deterministic mapping from a shared prefix ("Warbles...") to a
handful of memorized completions collides with a different, untrained
completion of that same prefix. It does - training on warbles made
held-out warble facts MUCH worse (measured delta -2.27 nats at Phase
0's own validated 200-step/1e-4 regime), while the exact same training
run improved Phase 0's quaddle probes by +0.71 nats - because quaddle
probes start with a different subject token entirely and never trigger
that collision. Two different mechanisms, same training run, opposite
signs.

So generalization is measured the way Phase 0 already validated it
works: against a separate SIBLING topic - same template/register,
different subject and vocabulary, never trained on at all. A topic has
exploitable structure if studying it measurably improves prediction on
its sibling; it's noise if it doesn't.

Three anchor categories:
- "known": real-world common-knowledge sentences, paired with a second,
  unrelated set of common-knowledge sentences as the sibling. Baseline
  loss should already be low for both.
- "moderate": Phase 0's original warbles (all 10 facts) paired with
  quaddles as the sibling - literally the same pair already validated
  in piper_assistant/feature/piper-memory, reused here rather than
  rebuilt.
- "noise": word-salad sentences, paired with a second word-salad set
  drawn from a completely disjoint vocabulary (verified programmatically,
  zero shared words) - the vocabulary-overlap leak an earlier version of
  this content had is exactly why that disjointness matters.
"""

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_PATH = Path(__file__).resolve().parent / "curiosity_topics.json"


def load_curiosity_topics(path: Path = DEFAULT_PATH) -> dict:
    topics = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_no_train_sibling_overlap(topics)
    return topics


def _validate_no_train_sibling_overlap(topics: dict) -> None:
    """Basic hygiene: a fact appearing in both train_facts and
    sibling_facts would mean the "sibling" isn't actually untouched by
    training, which defeats the whole point of using it as the
    generalization measurement."""
    for name, topic in topics.items():
        overlap = set(topic["train_facts"]) & set(topic["sibling_facts"])
        if overlap:
            raise ValueError(
                f"{name!r} has fact(s) in both train_facts and sibling_facts: {overlap} - "
                f"the sibling topic must never be trained on"
            )


def get_category(topics: dict, name: str) -> str:
    return topics[name]["category"]


def get_train_facts(topics: dict, name: str) -> List[str]:
    return topics[name]["train_facts"]


def get_trained_probes(topics: dict, name: str) -> List[Dict[str, str]]:
    """Memorization signal: probes decomposed from the exact facts
    trained on."""
    return topics[name]["trained_probes"]


def get_sibling_facts(topics: dict, name: str) -> List[str]:
    return topics[name]["sibling_facts"]


def get_sibling_probes(topics: dict, name: str) -> List[Dict[str, str]]:
    """Generalization signal: probes for a separate topic, same
    template/register, never trained on at all."""
    return topics[name]["sibling_probes"]
