"""
In-RAM, disk-persisted knowledge store.

Ported from piper_assistant's feature/piper-memory branch (src/piper_memory) -
see that repo's obsidian/Journals/2026-09-10.md and 2026-09-11.md for the
full design discussion and first real results. Built to answer a specific
need: a genuine learning-progress signal ("is studying this topic
measurably improving prediction, not just novel") requires a history of
repeated exposures to compute a trend from, and that history has to
survive process restarts to mean anything across sessions.

Deliberately NOT built on FAISS or any specialized vector index. At the
scale this is meant for (hundreds to low-thousands of items), brute-force
cosine similarity against an in-memory tensor is sub-millisecond - no
approximate-nearest-neighbor structure is needed, and keeping the whole
mechanism inside plain, inspectable code (rather than behind an external
library's own index format) was an explicit design choice, not just a
simplicity shortcut - see the "means of access" discussion in the journal.

Persistence is deliberately split rather than a single pickle: embeddings
via torch.save/load, metadata (content, timestamps, loss history) as JSON -
human-readable, diffable, and robust to this class's shape changing over
time, which pickle is not.

MemoryStore represents only what has actually been learned and retained.
Test/control material (e.g. a fictional entity deliberately never
studied) must never be added here - it belongs in a separate, static
test-definition file the evaluation harness reads directly, so a
negative control stays a negative control by construction.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MEMORY_DIR = REPO_ROOT / "data" / "memory"

METADATA_FILENAME = "metadata.json"
EMBEDDINGS_FILENAME = "embeddings.pt"

MEMORY_DIR_ENV_VAR = "CURIOUS_GEORGE_MEMORY_DIR"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_memory_dir() -> Path:
    """Resolved at call time, not import time, so a notebook can set
    CURIOUS_GEORGE_MEMORY_DIR (e.g. to a path under a mounted Google
    Drive) at any point before calling save()/load() with no explicit
    dir_path - the read/write session doesn't have to happen in the same
    cell as the import. Colab's local disk is wiped between runtime
    restarts, so this env var is how "start limited, learn and retain"
    actually survives across sessions there; the repo-relative
    DEFAULT_MEMORY_DIR remains the fallback for local/test use, where the
    working directory is already persistent."""
    override = os.environ.get(MEMORY_DIR_ENV_VAR)
    return Path(override) if override else DEFAULT_MEMORY_DIR


@dataclass
class MemoryItem:
    """One retained unit of knowledge. loss_history is a list of
    (timestamp, loss_value) pairs, append-only - never overwritten on a
    re-study, since the whole point is to keep every measurement so a
    trend can be computed from them later."""
    topic: str
    content: str
    first_learned: str
    last_studied: str
    loss_history: List[Tuple[str, float]] = field(default_factory=list)


class MemoryStore:
    """items is an ordered list, not just a dict, because embedding
    matrix rows are positionally aligned to it - item i's embedding under
    method m lives at embedding_matrices[m][i]. _topic_to_index gives
    O(1) lookup by topic without giving up that ordering."""

    def __init__(self):
        self.items: List[MemoryItem] = []
        self._topic_to_index: Dict[str, int] = {}
        self.embedding_matrices: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.items)

    def has_topic(self, topic: str) -> bool:
        return topic in self._topic_to_index

    def get_item(self, topic: str) -> Optional[MemoryItem]:
        idx = self._topic_to_index.get(topic)
        return self.items[idx] if idx is not None else None

    def add_or_update_item(self, topic: str, content: str, embeddings: Dict[str, torch.Tensor]) -> MemoryItem:
        """A new topic gets appended (to items and to every embedding
        matrix, keeping rows aligned) with fresh timestamps and an empty
        loss history. An existing topic is treated as a re-study: content
        and embeddings are refreshed in place, last_studied moves
        forward, but loss_history and first_learned are left untouched -
        re-studying isn't the same event as measuring loss, and first
        exposure shouldn't be forgotten just because content was
        refreshed."""
        now = _now_iso()
        idx = self._topic_to_index.get(topic)

        if idx is None:
            idx = len(self.items)
            self.items.append(MemoryItem(
                topic=topic, content=content, first_learned=now, last_studied=now, loss_history=[],
            ))
            self._topic_to_index[topic] = idx
            for method, vector in embeddings.items():
                row = vector.unsqueeze(0)
                if method in self.embedding_matrices:
                    self.embedding_matrices[method] = torch.cat([self.embedding_matrices[method], row], dim=0)
                else:
                    self.embedding_matrices[method] = row
        else:
            item = self.items[idx]
            item.content = content
            item.last_studied = now
            for method, vector in embeddings.items():
                if method not in self.embedding_matrices:
                    raise ValueError(
                        f"embedding method {method!r} was never registered on a prior item - "
                        f"all items must share the same set of embedding methods"
                    )
                self.embedding_matrices[method][idx] = vector

        return self.items[idx]

    def record_loss(self, topic: str, loss_value: float) -> None:
        idx = self._topic_to_index.get(topic)
        if idx is None:
            raise KeyError(f"cannot record loss for {topic!r} - it has never been studied (add_or_update_item first)")
        self.items[idx].loss_history.append((_now_iso(), loss_value))

    def learning_progress(self, topic: str) -> Optional[float]:
        """Positive = loss went down = improving. None if fewer than two
        measurements exist - a trend needs at least two points, and
        returning None rather than 0.0 keeps "not enough data yet"
        distinguishable from "measured no change."

        Simplest defensible version: first measurement minus most recent.
        A slope over every measurement (e.g. least-squares) would be more
        robust to a single noisy reading, but this is the version the
        design discussion settled on to start with - easy to swap in a
        more sophisticated trend later without changing the interface."""
        item = self.get_item(topic)
        if item is None:
            raise KeyError(f"no memory item for topic {topic!r}")
        if len(item.loss_history) < 2:
            return None
        first_loss = item.loss_history[0][1]
        latest_loss = item.loss_history[-1][1]
        return first_loss - latest_loss

    def query(self, query_vector: torch.Tensor, embedding_method: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Brute-force cosine similarity - deliberately not an
        approximate index, see module docstring. Returns (topic,
        similarity) pairs, highest similarity first. Empty store or an
        unregistered embedding method both return [] rather than
        raising - "nothing to find yet" is an expected, ordinary state
        for a store that starts empty by design."""
        matrix = self.embedding_matrices.get(embedding_method)
        if matrix is None or matrix.shape[0] == 0:
            return []
        similarities = torch.nn.functional.cosine_similarity(query_vector.unsqueeze(0), matrix, dim=1)
        k = min(top_k, similarities.shape[0])
        top_values, top_indices = torch.topk(similarities, k)
        return [(self.items[i].topic, top_values[j].item()) for j, i in enumerate(top_indices.tolist())]

    def save(self, dir_path: Optional[Path] = None) -> None:
        dir_path = Path(dir_path) if dir_path is not None else _default_memory_dir()
        dir_path.mkdir(parents=True, exist_ok=True)

        metadata = [
            {
                "topic": item.topic,
                "content": item.content,
                "first_learned": item.first_learned,
                "last_studied": item.last_studied,
                "loss_history": [[ts, loss] for ts, loss in item.loss_history],
            }
            for item in self.items
        ]
        (dir_path / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save(self.embedding_matrices, dir_path / EMBEDDINGS_FILENAME)

    @classmethod
    def load(cls, dir_path: Optional[Path] = None) -> "MemoryStore":
        """Missing files -> a fresh, empty store. First-ever run is a
        real, expected case here, not an error condition - the whole
        "start limited, learn and retain" design depends on that being
        true."""
        dir_path = Path(dir_path) if dir_path is not None else _default_memory_dir()
        metadata_path = dir_path / METADATA_FILENAME
        embeddings_path = dir_path / EMBEDDINGS_FILENAME

        store = cls()
        if not metadata_path.exists() or not embeddings_path.exists():
            return store

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embedding_matrices = torch.load(embeddings_path, weights_only=True)

        for idx, entry in enumerate(metadata):
            store.items.append(MemoryItem(
                topic=entry["topic"],
                content=entry["content"],
                first_learned=entry["first_learned"],
                last_studied=entry["last_studied"],
                loss_history=[tuple(pair) for pair in entry["loss_history"]],
            ))
            store._topic_to_index[entry["topic"]] = idx

        for method, matrix in embedding_matrices.items():
            if matrix.shape[0] != len(store.items):
                raise ValueError(
                    f"corrupt memory store: embedding method {method!r} has {matrix.shape[0]} rows "
                    f"but metadata has {len(store.items)} items - these must stay aligned"
                )
        store.embedding_matrices = embedding_matrices

        return store
