import pytest
import torch

from curious_george.curiosity_tools.study import study_topic
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

pytestmark = pytest.mark.slow  # loads and fine-tunes a real ~0.5B model on CPU


def test_study_topic_runs_a_real_session_and_records_into_loss_history():
    store = MemoryStore()
    content = "An autonomous agent must decide which of several competing goals to pursue right now."
    store.add_or_update_item("goal_arbitration", content, {"m": torch.tensor([1.0, 0.0])})
    store.record_loss("goal_arbitration", 4.2)  # the cheap-entry baseline reading
    store.set_status("goal_arbitration", PipelineStatus.ACTIVE)

    result = study_topic(store, MODEL_NAME, DEVICE, "goal_arbitration", num_steps=20, learning_rate=1e-3)

    for key in ("loss_before", "loss_after", "progress_this_session"):
        assert isinstance(result[key], float)
    assert result["progress_this_session"] > 0, (
        "a real study session should reduce cold loss on the topic's own content"
    )
    assert result["mastery"] in ("newbie", "novice", "master")

    item = store.get_item("goal_arbitration")
    assert len(item.loss_history) == 2, "the cheap-entry baseline plus this session's new reading"
    assert item.loss_history[-1][1] == result["loss_after"]
    assert item.last_studied != item.first_learned, "mark_studied should have moved last_studied forward"
