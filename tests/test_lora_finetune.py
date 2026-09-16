import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.curiosity_tools.lora_finetune import finetune_lora
from curious_george.curiosity_tools.loss_measurement import evaluate_recall_probes

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

pytestmark = pytest.mark.slow  # loads and fine-tunes a real ~0.5B model on CPU


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _fresh_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model


def test_base_model_weights_are_untouched_and_lora_params_move(tokenizer):
    model = _fresh_model()
    base_weight_before = model.model.layers[0].self_attn.q_proj.weight.clone().detach()

    toy_facts = [
        "Florbins are purple and silver insects.",
        "Florbins can burrow and glow but they cannot jump.",
    ]
    tuned_model = finetune_lora(model, tokenizer, DEVICE, toy_facts, num_steps=20, learning_rate=1e-3, r=4, verbose=False)

    base_weight_after = tuned_model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight
    assert torch.equal(base_weight_before, base_weight_after), (
        "the base model's own q_proj weight changed - LoRA should leave it byte-for-byte identical"
    )

    lora_params = [p for n, p in tuned_model.named_parameters() if "lora_" in n and p.requires_grad]
    assert len(lora_params) > 0, "expected trainable LoRA parameters to exist"
    assert any(torch.any(p != 0) for p in lora_params), (
        "expected at least one LoRA parameter to have moved away from its zero initialization"
    )


def test_loss_decreases_over_real_training_and_integrates_with_evaluate_recall_probes(tokenizer):
    toy_facts = [
        "Florbins are purple and silver insects.",
        "Florbins can burrow and glow but they cannot jump.",
    ]

    def measure_toy_loss(m):
        total = 0.0
        for fact in toy_facts:
            ids = tokenizer(fact, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
            with torch.no_grad():
                out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
            total += out.loss.item()
        return total / len(toy_facts)

    model = _fresh_model()
    pre_loss = measure_toy_loss(model)
    tuned_model = finetune_lora(model, tokenizer, DEVICE, toy_facts, num_steps=100, learning_rate=1e-3, r=4, verbose=False)
    post_loss = measure_toy_loss(tuned_model)

    assert post_loss < pre_loss, f"expected loss to decrease with real training: pre={pre_loss:.4f} post={post_loss:.4f}"

    probes = [{"prompt": "Florbins are purple and", "target": " silver insects."}]
    loss, accuracy = evaluate_recall_probes(tuned_model, tokenizer, DEVICE, probes)
    assert isinstance(loss, float) and isinstance(accuracy, float)
