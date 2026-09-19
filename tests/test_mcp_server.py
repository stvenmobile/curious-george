import torch

from curious_george.curiosity_tools import mcp_server
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus


def _seeded_store():
    store = MemoryStore()
    store.add_or_update_item("goal_arbitration", "An autonomous agent must decide which goal to pursue.",
                              {"small_model": torch.tensor([1.0, 0.0])})
    store.record_loss("goal_arbitration", 4.2)
    store.set_status("goal_arbitration", PipelineStatus.ACTIVE)
    return store


def test_list_topics_and_get_topic_serialize_without_a_live_model():
    # list_topics/get_topic only ever touch the store, never the warm
    # model/tokenizer/embedder - so they're testable without _startup()
    # having loaded anything, real or mocked.
    mcp_server._state["store"] = _seeded_store()

    listed = mcp_server.list_topics()
    assert len(listed) == 1
    assert listed[0]["topic"] == "goal_arbitration"
    assert listed[0]["status"] == "active"
    assert listed[0]["mastery"] == "novice"

    detail = mcp_server.get_topic("goal_arbitration")
    assert detail["content"] == "An autonomous agent must decide which goal to pursue."
    assert detail["loss_history"] == [[detail["loss_history"][0][0], 4.2]]


def test_list_topics_filters_by_status():
    store = _seeded_store()
    store.add_or_update_item("unrelated", "content", {"small_model": torch.tensor([0.0, 1.0])})
    mcp_server._state["store"] = store

    active_only = mcp_server.list_topics(status="active")
    assert [item["topic"] for item in active_only] == ["goal_arbitration"]


def test_get_topic_raises_for_unknown_topic():
    mcp_server._state["store"] = MemoryStore()
    try:
        mcp_server.get_topic("nonexistent")
        raised = False
    except KeyError:
        raised = True
    assert raised, "get_topic should raise for a topic that was never added"
