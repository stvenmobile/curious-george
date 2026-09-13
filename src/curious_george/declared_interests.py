"""
Declared interests: the human-seeded resonance anchors.

Autonomous taste-formation is out of scope for now (see
obsidian/Journals/2026-09-13.md) - these are topics we've told Piper
she's interested in, not topics she discovered on her own. Each
candidate topic's resonance score (MemoryStore.resonance) is its cosine
similarity to whichever declared interest it's closest to, computed via
the same embedding methods everything else in MemoryStore uses.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch

DEFAULT_PATH = Path(__file__).resolve().parent / "declared_interests.json"


def load_declared_interests(path: Path = DEFAULT_PATH) -> List[Dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def embed_declared_interests(interests: List[Dict[str, str]], embedder) -> Dict[str, torch.Tensor]:
    """Embeds each interest's description, not just its bare name - a
    short description gives the embedder more to work with than a
    two-or-three-word label, the same reason MemoryStore embeds whole
    sentences of content rather than bare topic names elsewhere."""
    return {entry["name"]: embedder.embed(entry["description"]) for entry in interests}
