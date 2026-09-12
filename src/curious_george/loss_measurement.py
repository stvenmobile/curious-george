"""
Teacher-forced loss measurement on recall probes.

The core "did studying this measurably help" signal. Deliberately not
tied to any particular way of changing the model's behavior - in-context
conditioning, LoRA fine-tuning, or nothing at all for a baseline
measurement - so the study mechanism can change later without this
needing to.

Just a normal single forward pass on one model, reading its own logits,
scoring only the target continuation's tokens.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def compute_probe_loss(model, tokenizer, device, prompt: str, target: str) -> Tuple[torch.Tensor, float]:
    """Teacher-forced cross-entropy on `target`, given `prompt` as
    context. Lower loss means the model predicts this specific
    continuation well - whether because it already knew it, or because
    something just taught it.

    Position accounting: logits at position prompt_len-1 predict the
    target's first token (pure prompt-based prediction, nothing else
    known yet - the most information-dense position); each later
    position's prediction also has the real preceding target tokens to
    work from (standard teacher forcing)."""
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prompt + target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]
    assert full_len > prompt_len, (
        f"target {target!r} added no tokens beyond the prompt {prompt!r} - nothing to score"
    )

    with torch.no_grad():
        outputs = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))

    prediction_logits = outputs.logits[0, prompt_len - 1: full_len - 1, :]
    targets = full_ids[0, prompt_len:full_len]

    loss = F.cross_entropy(prediction_logits.float(), targets)
    predicted_ids = prediction_logits.argmax(dim=-1)
    token_accuracy = (predicted_ids == targets).float().mean().item()
    return loss, token_accuracy


def evaluate_recall_probes(model, tokenizer, device, probes: List[Dict[str, str]]) -> Tuple[float, float]:
    """Averages compute_probe_loss over a whole entity's recall_probes -
    the "held-out loss" figure that gets recorded via
    MemoryStore.record_loss. Deliberately no_grad throughout - this is a
    read-only measurement, never a training step."""
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for probe in probes:
            loss, accuracy = compute_probe_loss(model, tokenizer, device, probe["prompt"], probe["target"])
            total_loss += loss.item()
            total_accuracy += accuracy
    n = len(probes)
    return total_loss / n, total_accuracy / n
