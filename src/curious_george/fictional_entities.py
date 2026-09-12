"""
Fabricated-knowledge validation content.

Real-world facts can never give an unambiguous signal for validating the
loss-delta learning-progress mechanism, because pretrained leakage can
never be ruled out - if performance improves, there's no way to know
whether the mechanism taught it or the model already half-knew it.
Fabricated facts about something that provably doesn't exist anywhere
else close that gap completely.

Two entities: "warbles" (role "studied" - the thing the mechanism
actually gets to learn about) and "quaddles" (role "control" - facts
that exist here only to define probes, and must NEVER be fed to a study
step). Quaddles is deliberately opposite warbles on most traits
(solitary vs. social, swim vs. fly, dawn vs. dusk), so any contamination
between the two would show up as a wrong-direction answer, not just an
ambiguous "did it improve a little."

Two kinds of probe: recall_probes are (prompt, target) continuation
pairs, meant for the fast, quantitative teacher-forced loss-delta
measurement - the primary learning-progress signal. generalization_probes
are open questions with no single correct continuation, meant for the
slower, free-generation qualitative read (does the answer actually
integrate multiple studied facts, or just recite one verbatim) - not part
of the loss-delta computation.
"""

import json
from pathlib import Path
from typing import List, Dict

DEFAULT_PATH = Path(__file__).resolve().parent / "fictional_entities.json"


def load_fictional_entities(path: Path = DEFAULT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_study_facts(entities: dict, name: str) -> List[str]:
    """Only entities with role == 'studied' may have their facts pulled
    for a study step. Raises for role == 'control' by construction,
    rather than relying on every caller remembering not to call this -
    keeping a negative control genuinely untouched is the entire point
    of having one, so this can't be an easy mistake to make."""
    entity = entities[name]
    if entity["role"] != "studied":
        raise ValueError(
            f"{name!r} has role={entity['role']!r}, not 'studied' - its facts must never be used "
            f"as study material, or it stops being a valid negative control"
        )
    return entity["study_facts"]


def get_recall_probes(entities: dict, name: str) -> List[Dict[str, str]]:
    """Available for any role - measuring loss on a control entity's
    probes (which should show no improvement) is exactly as important as
    measuring it on the studied entity's (which should)."""
    return entities[name]["recall_probes"]


def get_generalization_probes(entities: dict, name: str) -> List[str]:
    return entities[name]["generalization_probes"]
