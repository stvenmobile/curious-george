"""
My interests: the human-seeded resonance anchors.

Autonomous taste-formation is out of scope for now (see
obsidian/Journals/2026-09-13.md) - these are topics we've told Piper
she's interested in, not topics she discovered on her own. Each
candidate topic's resonance score (MemoryStore.resonance) is its cosine
similarity to whichever of "my interests" it's closest to, computed via
the same embedding methods everything else in MemoryStore uses.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

DEFAULT_PATH = Path(__file__).resolve().parent / "my_interests.json"

# A static, conceptual value for the protected 5 - not derived, never
# decays (see interest_pool.py's decaying open-slot entries). Always
# higher than the pool's STARTING_VALUE (60), so the protected 5 rank
# above anything in the open pool wherever the two are compared or
# displayed together. Chosen as a fixed number rather than infinity so
# it stays a real, comparable value instead of a special case.
PROTECTED_VALUE = 99.0


def load_my_interests(path: Path = DEFAULT_PATH) -> List[Dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_protected_interest(name: str, interests: Optional[List[Dict[str, str]]] = None) -> bool:
    interests = interests if interests is not None else load_my_interests()
    return any(entry["name"] == name for entry in interests)


def embed_my_interests(interests: List[Dict[str, str]], embedder) -> Dict[str, torch.Tensor]:
    """Embeds each interest's description, not just its bare name - a
    short description gives the embedder more to work with than a
    two-or-three-word label, the same reason MemoryStore embeds whole
    sentences of content rather than bare topic names elsewhere."""
    return {entry["name"]: embedder.embed(entry["description"]) for entry in interests}
