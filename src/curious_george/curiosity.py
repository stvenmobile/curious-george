"""
Phase 1: does baseline loss category predict actual learning progress?

The Phase 1 hypothesis, made testable: under a fixed, cheap LoRA trial
budget, a "moderate" topic should generalize what it studies to
held-out material about the same topic much better than a "noise"
topic does (which should only manage to memorize the exact sentences it
saw), while a "known" topic has little room to improve at all because
baseline loss is already low.

run_topic_trial reuses Phase 0's machinery unchanged - evaluate_recall_probes
and finetune_lora - it just runs the before/study/after pattern against
BOTH a trained-probe set (memorization) and a held-out-probe set
(generalization) instead of one shared set, and reports the ratio
between the two deltas as the actual noise measurement:

    generalization_progress / memorization_progress

Reloads the base model fresh for every topic rather than reusing one
peft-wrapped model across topics - avoids any risk of adapter state
leaking between trials, at the cost of a few extra seconds per topic to
reload already-cached weights.
"""

from typing import Dict, Iterable, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.curiosity_topics import (
    load_curiosity_topics, get_train_facts, get_trained_probes, get_held_out_probes, get_category,
)
from curious_george.loss_measurement import evaluate_recall_probes
from curious_george.lora_finetune import finetune_lora


def run_topic_trial(model_name: str, device: str, topic_name: str, topics: dict = None,
                     num_steps: int = 40, learning_rate: float = 1e-3) -> dict:
    """Runs one topic's cheap LoRA trial and returns both the
    memorization and generalization deltas. `topics` can be passed in
    (already loaded) so run_noise_experiment doesn't reload the JSON
    file once per topic; loads it itself if omitted, so this still
    works standalone."""
    if topics is None:
        topics = load_curiosity_topics()

    train_facts = get_train_facts(topics, topic_name)
    trained_probes = get_trained_probes(topics, topic_name)
    held_out_probes = get_held_out_probes(topics, topic_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()

    trained_loss_before, trained_acc_before = evaluate_recall_probes(base_model, tokenizer, device, trained_probes)
    heldout_loss_before, heldout_acc_before = evaluate_recall_probes(base_model, tokenizer, device, held_out_probes)

    tuned_model = finetune_lora(
        base_model, tokenizer, device, train_facts, num_steps=num_steps, learning_rate=learning_rate, verbose=False,
    )

    trained_loss_after, trained_acc_after = evaluate_recall_probes(tuned_model, tokenizer, device, trained_probes)
    heldout_loss_after, heldout_acc_after = evaluate_recall_probes(tuned_model, tokenizer, device, held_out_probes)

    memorization_progress = trained_loss_before - trained_loss_after
    generalization_progress = heldout_loss_before - heldout_loss_after
    generalization_ratio = (
        generalization_progress / memorization_progress if memorization_progress > 0 else None
    )

    return {
        "category": get_category(topics, topic_name),
        "trained_loss_before": trained_loss_before, "trained_loss_after": trained_loss_after,
        "trained_acc_before": trained_acc_before, "trained_acc_after": trained_acc_after,
        "heldout_loss_before": heldout_loss_before, "heldout_loss_after": heldout_loss_after,
        "heldout_acc_before": heldout_acc_before, "heldout_acc_after": heldout_acc_after,
        "memorization_progress": memorization_progress,
        "generalization_progress": generalization_progress,
        "generalization_ratio": generalization_ratio,
    }


def run_noise_experiment(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu",
                          num_steps: int = 40, learning_rate: float = 1e-3,
                          topic_names: Iterable[str] = ("known", "moderate", "noise")) -> Dict[str, dict]:
    """Runs run_topic_trial for every topic and prints a comparison
    table. topic_names defaults to all three anchor categories, but
    accepts a subset for a faster spot-check."""
    topics = load_curiosity_topics()
    results = {}

    for name in topic_names:
        print(f"[Curiosity] Running trial for topic '{name}' ({get_category(topics, name)})...")
        results[name] = run_topic_trial(model_name, device, name, topics=topics,
                                         num_steps=num_steps, learning_rate=learning_rate)

    print(f"\n{'topic':<10} {'category':<10} {'baseline loss':>14} {'memorization':>13} {'generalization':>15} {'gen/mem ratio':>14}")
    for name, r in results.items():
        baseline_loss = (r["trained_loss_before"] + r["heldout_loss_before"]) / 2
        ratio_str = f"{r['generalization_ratio']:.3f}" if r["generalization_ratio"] is not None else "n/a"
        print(f"{name:<10} {r['category']:<10} {baseline_loss:>14.4f} {r['memorization_progress']:>13.4f} "
              f"{r['generalization_progress']:>15.4f} {ratio_str:>14}")

    return results


if __name__ == "__main__":
    run_noise_experiment()
