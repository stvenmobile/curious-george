import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.lora_finetune import finetune_lora, _average_facts_loss

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

pytestmark = pytest.mark.slow  # loads and fine-tunes a real ~0.5B model on CPU

TOY_FACTS = [
    "Florbins are purple and silver insects.",
    "Florbins can burrow and glow but they cannot jump.",
    "Florbins live inside old stone walls.",
]


@pytest.fixture(scope="module")
def tuned_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return finetune_lora(model, tokenizer, DEVICE, TOY_FACTS, num_steps=30, learning_rate=1e-3, r=4, verbose=False)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def test_matches_independently_computed_manual_average(tuned_model, tokenizer):
    manual_total = 0.0
    tuned_model.eval()
    with torch.no_grad():
        for fact in TOY_FACTS:
            ids = tokenizer(fact, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
            out = tuned_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
            manual_total += out.loss.item()
    manual_avg = manual_total / len(TOY_FACTS)

    reported_avg = _average_facts_loss(tuned_model, tokenizer, DEVICE, TOY_FACTS)
    assert abs(reported_avg - manual_avg) < 1e-6, f"expected {manual_avg}, got {reported_avg}"


def test_restores_train_mode_when_called_mid_training(tuned_model, tokenizer):
    tuned_model.train()
    assert tuned_model.training is True
    _ = _average_facts_loss(tuned_model, tokenizer, DEVICE, TOY_FACTS)
    assert tuned_model.training is True, "must restore train() mode if it was training before the call"


def test_leaves_eval_mode_alone_when_called_outside_training(tuned_model, tokenizer):
    tuned_model.eval()
    assert tuned_model.training is False
    _ = _average_facts_loss(tuned_model, tokenizer, DEVICE, TOY_FACTS)
    assert tuned_model.training is False, "must leave eval() mode alone if it was already in eval mode"
