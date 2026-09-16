import torch

from curious_george.curiosity_tools.study import study_topic
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus


def test_study_topic_raises_for_unknown_topic():
    store = MemoryStore()
    try:
        study_topic(store, "fake-model", "cpu", "nonexistent")
        raised = False
    except KeyError:
        raised = True
    assert raised, "studying a topic that was never added should raise, not silently no-op"


def test_study_topic_raises_for_non_active_status():
    store = MemoryStore()
    store.add_or_update_item("candidate_topic", "content", {"m": torch.tensor([1.0, 0.0])})
    # status defaults to CANDIDATE, never promoted to ACTIVE

    try:
        study_topic(store, "fake-model", "cpu", "candidate_topic")
        raised = False
    except ValueError:
        raised = True
    assert raised, "real study should refuse a topic that never survived deep-scoring"


def test_study_topic_raises_for_archived_status():
    store = MemoryStore()
    store.add_or_update_item("archived_topic", "content", {"m": torch.tensor([1.0, 0.0])})
    store.set_status("archived_topic", PipelineStatus.ARCHIVED)

    try:
        study_topic(store, "fake-model", "cpu", "archived_topic")
        raised = False
    except ValueError:
        raised = True
    assert raised, "real study should refuse an archived (already mastered) topic too"
