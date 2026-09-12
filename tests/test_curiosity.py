import pytest

from curious_george.curiosity import run_topic_trial

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

pytestmark = pytest.mark.slow  # loads and fine-tunes a real ~0.5B model on CPU


def test_run_topic_trial_returns_well_formed_results_for_moderate():
    """Plumbing check, not a hypothesis check: this only asserts things
    Phase 0 already established (LoRA reduces loss on what it's
    directly trained on). Whether moderate's held-out generalization
    beats noise's is the actual open question for the real multi-topic
    run - not something to bake in as a pass/fail assertion here."""
    result = run_topic_trial(MODEL_NAME, DEVICE, "moderate", num_steps=20, learning_rate=1e-3)

    for key in ("trained_loss_before", "trained_loss_after", "heldout_loss_before", "heldout_loss_after",
                "memorization_progress", "generalization_progress"):
        assert isinstance(result[key], float)

    assert result["category"] == "moderate"
    assert result["memorization_progress"] > 0, (
        "training directly on these facts should reduce loss on probes decomposed from those same facts"
    )


def test_run_topic_trial_works_for_known_and_noise_categories_too():
    """Just confirms the harness runs end-to-end for the other two
    categories without erroring - no outcome assertions, since the
    whole point of those categories is that the outcome is genuinely
    unknown until measured."""
    known_result = run_topic_trial(MODEL_NAME, DEVICE, "known", num_steps=20, learning_rate=1e-3)
    noise_result = run_topic_trial(MODEL_NAME, DEVICE, "noise", num_steps=20, learning_rate=1e-3)

    assert known_result["category"] == "known"
    assert noise_result["category"] == "noise"
    for result in (known_result, noise_result):
        assert isinstance(result["generalization_progress"], float)
