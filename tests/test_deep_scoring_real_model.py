import pytest

from curious_george.curiosity_tools.deep_scoring import deep_score_trial, deep_score_topic
from curious_george.curiosity_tools.canary_topics import load_canary_topics, all_canary_probes

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

pytestmark = pytest.mark.slow  # loads and fine-tunes a real ~0.5B model on CPU


def test_deep_score_trial_returns_well_formed_results_against_the_real_model():
    """Plumbing check: confirms the wiring (measure_content_loss +
    finetune_lora + evaluate_recall_probes against the canary pool) runs
    end to end and returns sane types. Not a hypothesis check - what a
    real candidate's ratio comes out to is an open question, not
    something to assert here."""
    canary_probes = all_canary_probes(load_canary_topics())
    result = deep_score_trial(
        MODEL_NAME, DEVICE,
        content="An autonomous agent must decide which of several competing goals to pursue right now.",
        canary_probes=canary_probes, num_steps=20, learning_rate=1e-3,
    )

    for key in ("content_loss_before", "content_loss_after", "canary_loss_before", "canary_loss_after",
                "memorization_progress", "generalization_progress"):
        assert isinstance(result[key], float)
    assert result["memorization_progress"] > 0, (
        "training directly on this content should reduce loss on that same content"
    )


def test_deep_score_topic_averages_multiple_trials_into_one_record():
    record = deep_score_topic(
        MODEL_NAME, DEVICE,
        content="A short passage about deciding what to research next.",
        num_trials=2, num_steps=20, learning_rate=1e-3,
    )
    assert record.n_trials == 2
    assert isinstance(record.memorization_mean, float)
    assert isinstance(record.generalization_mean, float)
    assert record.timestamp
