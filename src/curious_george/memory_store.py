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

One store, not two. An item that's merely been noticed (cheap-scored:
one baseline loss reading plus an embedding) lives right alongside one
that's been thoroughly studied - there's no separate "candidate"
registry. See obsidian/Journals/2026-09-13.md for the design discussion
this schema implements: mastery and resonance are both derived from
data already on the item (loss_history and its embedding respectively,
never stored redundantly), but deep-scoring (transferability - the
signal that actually distinguishes learnable structure from noise, see
notebooks/curious_george.ipynb's Phase 1 result) is expensive and DOES
need its own history, since it can't be cheaply recomputed on demand the
way mastery/resonance can.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MEMORY_DIR = REPO_ROOT / "data" / "memory"

METADATA_FILENAME = "metadata.json"
EMBEDDINGS_FILENAME = "embeddings.pt"

MEMORY_DIR_ENV_VAR = "CURIOUS_GEORGE_MEMORY_DIR"

# Mastery thresholds, anchored to real numbers already measured rather
# than picked arbitrarily: "master" matches warbles post-study
# (~0.72-0.86 nats) and known-facts' baseline (~1.26 nats); "novice"
# matches warbles pre-study (~3.47 nats) - recognizably novel but not
# opaque; anything at or above "newbie" matches noise's baseline
# (~7.8-9.1 nats) - near the pure-guessing ceiling.
MASTERY_LOSS_CEILING = 1.5
NOVICE_LOSS_CEILING = 6.0


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


def classify_mastery(loss: float) -> str:
    """Newbie/novice/master, purely a function of the current loss
    value - see the module-level threshold constants for where these
    boundaries come from. Kept as a standalone function (not baked into
    MemoryItem) so it can be reused wherever a raw loss number needs a
    human-readable label, not just via MemoryStore.mastery_level."""
    if loss < MASTERY_LOSS_CEILING:
        return "master"
    if loss < NOVICE_LOSS_CEILING:
        return "novice"
    return "newbie"


class PipelineStatus(str, Enum):
    """Where an item currently sits in the interest-selection lifecycle -
    NOT a measure of understanding (that's mastery, derived from loss)
    and NOT a measure of interest (that's resonance, derived from the
    item's own embedding) - just its selection eligibility.

    CANDIDATE - cheap-scored only (one baseline loss reading, plus an
    embedding); never deep-scored, not yet competing for idle-time
    selection.
    ACTIVE - deep-scored at least once; currently eligible for
    idle-time selection.
    ARCHIVED - mastered, with no new information sources currently
    available to push further; kept, but no longer competing.

    Deliberately no PRUNED value here - an item whose deep score reveals
    it's noise doesn't get relabeled, it gets removed from the store
    entirely (see MemoryStore.remove_item). Marking-then-removing would
    just be two steps where one does the job."""
    CANDIDATE = "candidate"
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass
class DeepScoreRecord:
    """One deep-scoring event. Already an average across several LoRA
    trials (see curiosity.run_repeated_trials/summarize_trials), not a
    single raw trial - Phase 1 measured real run-to-run variance
    directly (moderate's ratio ranged 0.126-0.322 across 5 seeds), so a
    single-trial deep score isn't trustworthy enough to prune or select
    on. Kept as a history (append-only, like loss_history), not
    overwritten, so a rescore after further study can be compared
    against its own prior deep score to see whether a topic is nearing
    its mastery limit."""
    timestamp: str
    n_trials: int
    memorization_mean: float
    generalization_mean: float
    generalization_ratio_mean: Optional[float]
    generalization_ratio_stdev: Optional[float]


@dataclass
class MemoryItem:
    """One retained unit of knowledge. loss_history is a list of
    (timestamp, loss_value) pairs, append-only - never overwritten on a
    re-study, since the whole point is to keep every measurement so a
    trend can be computed from them later. deep_score_history is the
    same append-only pattern, applied to the expensive transferability
    signal instead of the cheap loss signal."""
    topic: str
    content: str
    first_learned: str
    last_studied: str
    loss_history: List[Tuple[str, float]] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.CANDIDATE
    deep_score_history: List[DeepScoreRecord] = field(default_factory=list)


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
        matrix, keeping rows aligned) with fresh timestamps, an empty
        loss history, and CANDIDATE status. An existing topic is treated
        as a re-study: content and embeddings are refreshed in place,
        last_studied moves forward, but loss_history, status, and
        deep_score_history are left untouched - re-studying isn't the
        same event as measuring loss or deep-scoring, and none of that
        history should be forgotten just because content was refreshed."""
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

    def mastery_level(self, topic: str) -> Optional[str]:
        """newbie/novice/master from the item's most recent loss
        reading. None if the item has never had a loss measurement at
        all yet (shouldn't normally happen - even a cheap entry-scoring
        pass records one baseline reading - but a fresh add_or_update_item
        with no record_loss call yet is a real, momentary state)."""
        item = self.get_item(topic)
        if item is None:
            raise KeyError(f"no memory item for topic {topic!r}")
        if not item.loss_history:
            return None
        return classify_mastery(item.loss_history[-1][1])

    def resonance(self, topic: str, embedding_method: str, interest_embeddings: Dict[str, torch.Tensor]) -> Optional[float]:
        """Cosine similarity between this item's already-stored
        embedding and whichever declared interest it's closest to.
        Deliberately not a stored field - the item's embedding already
        lives in embedding_matrices, and recomputing this on demand from
        it means resonance can never go stale if the declared-interest
        list changes later. Returns None if the item has no embedding
        under embedding_method, or if interest_embeddings is empty."""
        idx = self._topic_to_index.get(topic)
        if idx is None:
            raise KeyError(f"no memory item for topic {topic!r}")
        matrix = self.embedding_matrices.get(embedding_method)
        if matrix is None or not interest_embeddings:
            return None

        item_vector = matrix[idx].unsqueeze(0)
        best_similarity = None
        for interest_vector in interest_embeddings.values():
            similarity = torch.nn.functional.cosine_similarity(item_vector, interest_vector.unsqueeze(0)).item()
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
        return best_similarity

    def record_deep_score(self, topic: str, record: DeepScoreRecord) -> None:
        idx = self._topic_to_index.get(topic)
        if idx is None:
            raise KeyError(f"cannot record a deep score for {topic!r} - it has never been added (add_or_update_item first)")
        self.items[idx].deep_score_history.append(record)

    def latest_deep_score(self, topic: str) -> Optional[DeepScoreRecord]:
        item = self.get_item(topic)
        if item is None:
            raise KeyError(f"no memory item for topic {topic!r}")
        return item.deep_score_history[-1] if item.deep_score_history else None

    def set_status(self, topic: str, status: PipelineStatus) -> None:
        idx = self._topic_to_index.get(topic)
        if idx is None:
            raise KeyError(f"no memory item for topic {topic!r}")
        self.items[idx].status = status

    def remove_item(self, topic: str) -> None:
        """Permanently removes an item and its embedding rows - e.g.
        after a deep-scoring pass concludes it's noise. Re-aligns every
        embedding matrix and rebuilds _topic_to_index so items after the
        removed one shift down correctly; leaving a gap would break the
        positional alignment every other method depends on."""
        idx = self._topic_to_index.get(topic)
        if idx is None:
            raise KeyError(f"no memory item for topic {topic!r}")

        del self.items[idx]
        for method, matrix in self.embedding_matrices.items():
            self.embedding_matrices[method] = torch.cat([matrix[:idx], matrix[idx + 1:]], dim=0)
        self._topic_to_index = {item.topic: i for i, item in enumerate(self.items)}

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
                "status": item.status.value,
                "deep_score_history": [asdict(record) for record in item.deep_score_history],
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
        true. status/deep_score_history are read with .get(...) and a
        default, so a store saved before this schema extension existed
        still loads correctly - every item just starts as CANDIDATE with
        no deep-score history, which is the honest state for pre-existing
        items that were never deep-scored under this mechanism."""
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
                status=PipelineStatus(entry.get("status", PipelineStatus.CANDIDATE.value)),
                deep_score_history=[DeepScoreRecord(**record) for record in entry.get("deep_score_history", [])],
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
