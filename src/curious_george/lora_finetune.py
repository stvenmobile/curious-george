"""
LoRA fine-tuning - the genuine "study" step.

Where in-context "study" just places facts in the prompt (and was shown,
with real numbers, to produce a confounded signal - generic context
priming, not fact-specific learning), this actually changes the model's
own behavior via a small set of trained weights, so a COLD post-study
measurement (no facts anywhere in the prompt) becomes possible - the
only way to rule out generic context-priming and get a signal that
reflects persistent, fact-specific learning.

Via HuggingFace's peft library: freeze the base model entirely, inject
small trainable low-rank matrices alongside the attention q_proj/v_proj
layers instead of training them directly. The base model's own weights
never change at all - the result is base-model-plus-a-swappable-adapter,
non-destructive by construction.

target_modules=["q_proj", "v_proj"] is the standard, most common minimal
LoRA configuration - confirmed against Qwen2's actual attention-block
module names rather than assumed from memory.
"""

import random
from typing import List

import torch
from peft import LoraConfig, TaskType, get_peft_model


def _average_facts_loss(peft_model, tokenizer, device, study_facts: List[str]) -> float:
    """Average causal-LM loss across every study fact, under no_grad - a
    diagnostic eval, not a training step. Used for progress reporting so
    the printed curve reflects overall progress across the whole study
    set, not just whichever single fact rng.choice happened to sample
    that step - a real run showed exactly why that distinction matters:
    the per-step-only printout bounced between 0.27 and 3.5 across 200
    steps, which looked like training instability but was actually just
    reporting granularity - different facts sit at different loss values
    at any given moment, so which one gets sampled dominates the printed
    number far more than overall progress does."""
    was_training = peft_model.training
    peft_model.eval()
    total = 0.0
    with torch.no_grad():
        for fact in study_facts:
            ids = tokenizer(fact, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            outputs = peft_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
            total += outputs.loss.item()
    if was_training:
        peft_model.train()
    return total / len(study_facts)


def finetune_lora(model, tokenizer, device, study_facts: List[str],
                   num_steps: int = 200, learning_rate: float = 1e-4,
                   r: int = 8, seed: int = 1234, verbose: bool = True):
    """Wraps `model` with a trainable LoRA adapter and fine-tunes it on
    `study_facts` via standard causal-LM loss (predict each token from
    the ones before it, across the whole fact sentence - no prompt/target
    split needed here, unlike compute_probe_loss, since the goal is for
    the model to genuinely learn the fact, not just complete a fixed
    prefix). Returns the wrapped model, ready for inference.

    Skip (not crash on) a non-finite loss rather than let it poison
    Adam's moment estimates, and clip gradient norms as a margin against
    smaller spikes."""
    config = LoraConfig(
        r=r, lora_alpha=r * 2, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, config)
    peft_model.train()

    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=learning_rate,
    )
    rng = random.Random(seed)
    skipped_steps = 0
    last_loss = None

    for step in range(1, num_steps + 1):
        fact = rng.choice(study_facts)
        ids = tokenizer(fact, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        optimizer.zero_grad()
        outputs = peft_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        loss = outputs.loss

        if not torch.isfinite(loss):
            skipped_steps += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in peft_model.parameters() if p.requires_grad), max_norm=1.0,
        )
        optimizer.step()
        last_loss = loss.item()

        if verbose and (step % max(1, num_steps // 10) == 0 or step == num_steps):
            avg_loss = _average_facts_loss(peft_model, tokenizer, device, study_facts)
            print(f"[LoRA] step {step}/{num_steps}  last_step_loss={last_loss:.4f}  "
                  f"avg_loss_all_facts={avg_loss:.4f}"
                  + (f"  ({skipped_steps} skipped so far)" if skipped_steps else ""))

    peft_model.eval()
    return peft_model
