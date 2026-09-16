"""
The bounded, self-curating pool of resonance anchors beyond the
original 5 protected interests.

TOTAL effective capacity across BOTH sources is 10: 5 static, protected
interests assigned externally as initial research goals
(my_interests.json/.py - this module never touches them, they never
decay), plus OPEN_SLOT_CAPACITY (5) dynamic slots managed here - room
for candidates to surface either as external "suggestions" or, down the
road, from something Piper generates internally during her own research
activity. Nothing generates that second kind yet; the capacity is
reserved for it regardless, per obsidian/Journals/2026-09-16.md.

Anything added here starts at STARTING_VALUE (60) and loses
DECAY_PER_DAY (1) point per day since its last real study event (not
merely being matched/considered) - roughly two months to fully decay to
zero if never studied again. A new candidate can only displace the
weakest current entry if it currently outranks it - a fresh arrival
(value 60) beats anything that has decayed at all, and ties (both at
60, e.g. both added/studied today) are not evicted.

The promotion trigger is deliberately NOT automatic. There's no
autonomous pipeline process calling add_interest - for now it's called
manually (by the user or in an agent session), e.g. to test how
core-vs-later-added interests behave differently. That's a real,
deliberate choice, not a placeholder for something more automatic
coming soon - see the journal.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POOL_DIR = REPO_ROOT / "data" / "interest_pool"
POOL_FILENAME = "pool.json"

STARTING_VALUE = 60.0
DECAY_PER_DAY = 1.0
OPEN_SLOT_CAPACITY = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_between(earlier_iso: str, later_iso: str) -> float:
    earlier = datetime.fromisoformat(earlier_iso)
    later = datetime.fromisoformat(later_iso)
    return (later - earlier).total_seconds() / 86400.0


@dataclass
class PoolEntry:
    """anchor_since is reset to "now" whenever record_study fires - it's
    not a creation timestamp, it's "the last time real study activity
    happened," which is what current_value actually counts decay from."""
    name: str
    description: str
    anchor_since: str


def current_value(entry: PoolEntry, now: Optional[str] = None) -> float:
    now = now or _now_iso()
    days_idle = _days_between(entry.anchor_since, now)
    return max(0.0, STARTING_VALUE - DECAY_PER_DAY * days_idle)


def interest_value(name: str, pool: "InterestPool", now: Optional[str] = None) -> float:
    """Unified value lookup across BOTH sources - my_interests.py's
    PROTECTED_VALUE (99, static) for one of the original 5, else this
    pool's decaying current_value. Raises if `name` is neither, since
    that means it isn't a real interest at all under either source."""
    from curious_george.curiosity_tools.my_interests import PROTECTED_VALUE, is_protected_interest

    if is_protected_interest(name):
        return PROTECTED_VALUE
    entry = pool.get(name)
    if entry is None:
        raise KeyError(f"{name!r} is neither a protected interest nor in the pool")
    return current_value(entry, now)


class InterestPool:
    def __init__(self):
        self.entries: List[PoolEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, name: str) -> Optional[PoolEntry]:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def add_interest(self, name: str, description: str, now: Optional[str] = None) -> bool:
        """Manually promotes a new open-slot interest. Returns whether it
        was actually added - True if there was a free slot, or if it
        beat the weakest current entry at capacity; False if the pool is
        full and nothing was weak enough to displace."""
        if self.get(name) is not None:
            raise ValueError(f"{name!r} is already in the interest pool")
        now = now or _now_iso()
        new_entry = PoolEntry(name=name, description=description, anchor_since=now)

        if len(self.entries) < OPEN_SLOT_CAPACITY:
            self.entries.append(new_entry)
            return True

        weakest = min(self.entries, key=lambda e: current_value(e, now))
        if current_value(new_entry, now) > current_value(weakest, now):
            self.entries.remove(weakest)
            self.entries.append(new_entry)
            return True
        return False

    def record_study(self, name: str, now: Optional[str] = None) -> None:
        """Real study happened on this interest - resets its decay clock
        back to full value. The protected 5 never need this; they're not
        tracked here at all."""
        entry = self.get(name)
        if entry is None:
            raise KeyError(f"{name!r} is not in the interest pool - add_interest first")
        entry.anchor_since = now or _now_iso()

    def all_entries(self) -> List[Dict[str, str]]:
        """{"name", "description"} shape, matching my_interests.py's
        loader - lets a caller combine both sources into one resonance
        anchor list without needing to know this module's internal
        PoolEntry shape."""
        return [{"name": e.name, "description": e.description} for e in self.entries]

    def save(self, dir_path: Path = DEFAULT_POOL_DIR) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        data = [asdict(entry) for entry in self.entries]
        (dir_path / POOL_FILENAME).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Path = DEFAULT_POOL_DIR) -> "InterestPool":
        dir_path = Path(dir_path)
        path = dir_path / POOL_FILENAME
        pool = cls()
        if not path.exists():
            return pool
        data = json.loads(path.read_text(encoding="utf-8"))
        pool.entries = [PoolEntry(**entry) for entry in data]
        return pool
