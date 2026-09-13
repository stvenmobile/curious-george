"""
Canary siblings: a fixed generalization-quality check for real
candidates.

Phase 1 validated transferability against a domain-MATCHED sibling
(warbles/quaddles) - proving the model learned a reusable template, not
just memorized specific words. A real pipeline candidate (e.g. "goal
arbitration") doesn't come with a hand-built, domain-matched sibling,
and building one per candidate is impractical - see
obsidian/Journals/2026-09-14.md for the discussion.

These three topics exist instead as a small, fixed set of "canaries":
genuinely offbeat, unrelated domains (lighthouse keeping, vintage
typewriters, traditional knots) that real research candidates are
unlikely to thematically overlap with. Deep-scoring a real candidate
measures loss on the COMBINED pool of all three canaries' probes, not
just one - reduces the risk that a single accidental thematic overlap
between a candidate and one canary dominates the result.

Important: this measures something related but different from Phase 1's
original generalization_ratio. Not "did this candidate's specific
structure transfer to a matched partner" - that requires domain-matched
siblings we don't have for arbitrary content - but "did fine-tuning on
this candidate produce a well-behaved update, or a narrow, corrupting
overfit." A generalization-quality canary, not a structural-transfer
probe, even though it reuses the same before/study/after mechanics.
"""

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_PATH = Path(__file__).resolve().parent / "canary_topics.json"


def load_canary_topics(path: Path = DEFAULT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_canary_probes(topics: dict, name: str) -> List[Dict[str, str]]:
    return topics[name]["probes"]


def all_canary_probes(topics: dict) -> List[Dict[str, str]]:
    """The combined pool across every canary topic - what deep_scoring
    actually measures against, rather than any single canary alone."""
    combined = []
    for topic in topics.values():
        combined.extend(topic["probes"])
    return combined
