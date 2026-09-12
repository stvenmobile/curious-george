"""
Phase 1: the memorization-vs-generalization topic bank.

Phase 0 proved LoRA produces genuine, cold, persistent learning - but
its recall_probes were always decompositions of the exact sentences
trained on (e.g. study_facts has "Warbles are green and yellow
mammals." and the matching recall_probe is that same sentence split
into a prompt and target). That measures memorization. It has never
measured whether the model generalizes the underlying pattern to
material it was never shown - which is the actual distinction between
"vaguely understood, learnable structure" (this project's working
definition of what curiosity should target) and "high loss because
there's nothing there to learn" (noise).

Each topic below is split into train_facts (what the LoRA trial step
studies) and held_out_facts (same topic, same template, NEVER included
in train_facts). trained_probes measures memorization; held_out_probes
measures generalization - whether studying the trained facts made the
model any better at completing DIFFERENT facts about the same topic it
was never shown. The gap between those two deltas is the actual, usable
measurement of how much exploitable structure a topic has:

    generalization_progress / memorization_progress

near 0 means the model could only memorize what it saw (noise); a ratio
close to 1 means studying transferred almost as well to unseen material
as it worked on the trained material itself (real structure).

Three categories, deliberately spanning the baseline-loss range the
Phase 1 hypothesis is actually about:

- "known": real-world common-knowledge sentences. Baseline loss should
  already be low - the model has seen this many times. Included as the
  ceiling-effect anchor: little room to improve, whatever the
  generalization ratio turns out to be.
- "moderate": the warble facts from Phase 0, repartitioned - the 8
  facts that already had a matching recall_probe become train_facts,
  and the 2 that never had one (deliberately, in the original content)
  become held_out_facts. Novel surface form, but every fact follows the
  same slot template (subject + one fixed trait category per
  sentence), which is exactly the kind of structure a pattern should be
  able to transfer from.
- "noise": word-salad sentences with no shared template, no consistent
  subject, and no cross-sentence relationship beyond incidental
  vocabulary reuse. Nothing here should transfer, by construction - the
  point is to see whether the measurement actually shows that, not to
  assume it.
"""

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_PATH = Path(__file__).resolve().parent / "curiosity_topics.json"


def load_curiosity_topics(path: Path = DEFAULT_PATH) -> dict:
    topics = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_no_train_held_out_overlap(topics)
    return topics


def _validate_no_train_held_out_overlap(topics: dict) -> None:
    """A held-out fact that also appears in train_facts isn't held out -
    it silently turns the generalization measurement back into a
    memorization measurement. Checked at load time, not left to
    caller discipline, for the same reason get_study_facts raises for a
    control-role entity in fictional_entities.py."""
    for name, topic in topics.items():
        overlap = set(topic["train_facts"]) & set(topic["held_out_facts"])
        if overlap:
            raise ValueError(
                f"{name!r} has fact(s) listed in both train_facts and held_out_facts: {overlap} - "
                f"a held-out fact that was also trained on invalidates the generalization measurement"
            )


def get_category(topics: dict, name: str) -> str:
    return topics[name]["category"]


def get_train_facts(topics: dict, name: str) -> List[str]:
    return topics[name]["train_facts"]


def get_trained_probes(topics: dict, name: str) -> List[Dict[str, str]]:
    """Memorization signal: probes decomposed from the exact facts
    trained on."""
    return topics[name]["trained_probes"]


def get_held_out_facts(topics: dict, name: str) -> List[str]:
    return topics[name]["held_out_facts"]


def get_held_out_probes(topics: dict, name: str) -> List[Dict[str, str]]:
    """Generalization signal: probes decomposed from facts about the
    same topic that were NEVER included in train_facts."""
    return topics[name]["held_out_probes"]
