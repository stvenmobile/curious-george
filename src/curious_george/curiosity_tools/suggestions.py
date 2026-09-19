"""
Suggestions: the human-curated candidate source for the RESEARCH trigger.

Candidate generation is deliberately NOT autonomous (see README.md's
non-goals) - every candidate the RESEARCH trigger considers comes from
this file, hand-edited the same way my_interests.json and
canary_topics.json are: a durable, version-controlled record of what's
been proposed, not a runtime-only queue.

Schema: a JSON list of {"topic": ..., "content": ..., "note": ...}
objects. "note" is optional and purely for the human editor's own
reference - it's never read by any code here.
"""

import json
from pathlib import Path
from typing import Dict, List

from curious_george.curiosity_tools.memory_store import MemoryStore

DEFAULT_PATH = Path(__file__).resolve().parent / "suggestions.json"


def load_suggestions(path: Path = DEFAULT_PATH) -> List[Dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_suggestions(store: MemoryStore, suggestions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filters out any suggestion whose topic is already in the store
    under ANY status - candidate, active, shelved, or archived. A
    suggestion that's already been proposed once shouldn't be re-added
    just because it's still sitting in suggestions.json; re-studying an
    existing topic goes through study_topic/add_or_update_item's own
    "re-study" path instead, not through this ingestion step."""
    return [s for s in suggestions if not store.has_topic(s["topic"])]
