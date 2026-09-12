"""
Phase 1: does baseline loss category predict actual learning progress?

run_topic_trial measures loss on a topic's own trained_probes
(memorization) and on a separate sibling topic's probes (generalization -
same template/register, different subject, never trained on) before and
after a LoRA study step on train_facts only. generalization_progress /
memorization_progress is the actual noise measurement: near 0 means
studying didn't transfer anywhere, positive means it did.

Defaults to num_steps=200, learning_rate=1e-4 - Phase 0's own validated
regime (run_lora_check), not a "cheap trial" budget. An earlier version
of this experiment used a much cheaper budget (40 steps, 1e-3) and it
mattered less than expected: the real problem that run exposed was a
content design flaw (same-topic held-out facts colliding with trained
material), not the step budget. Whether a genuinely cheap trial predicts
the full-budget result is still an open question for later, not
something to assume by using a cheap default here.

Reloads the base model fresh for every topic rather than reusing one
peft-wrapped model across topics - avoids any risk of adapter state
leaking between trials, at the cost of a few extra seconds per topic to
reload already-cached weights.
"""

from typing import Dict, Iterable

from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.curiosity_topics import (
    load_curiosity_topics, get_train_facts, get_trained_probes, get_sibling_probes, get_category,
)
from curious_george.loss_measurement import evaluate_recall_probes
from curious_george.lora_finetune import finetune_lora


def run_topic_trial(model_name: str, device: str, topic_name: str, topics: dict = None,
                     num_steps: int = 200, learning_rate: float = 1e-4) -> dict:
    """Runs one topic's LoRA study step and returns both the
    memorization and generalization (sibling-topic) deltas. `topics`
    can be passed in (already loaded) so run_noise_experiment doesn't
    reload the JSON file once per topic; loads it itself if omitted, so
    this still works standalone."""
    if topics is None:
        topics = load_curiosity_topics()

    train_facts = get_train_facts(topics, topic_name)
    trained_probes = get_trained_probes(topics, topic_name)
    sibling_probes = get_sibling_probes(topics, topic_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()

    trained_loss_before, trained_acc_before = evaluate_recall_probes(base_model, tokenizer, device, trained_probes)
    sibling_loss_before, sibling_acc_before = evaluate_recall_probes(base_model, tokenizer, device, sibling_probes)

    tuned_model = finetune_lora(
        base_model, tokenizer, device, train_facts, num_steps=num_steps, learning_rate=learning_rate, verbose=False,
    )

    trained_loss_after, trained_acc_after = evaluate_recall_probes(tuned_model, tokenizer, device, trained_probes)
    sibling_loss_after, sibling_acc_after = evaluate_recall_probes(tuned_model, tokenizer, device, sibling_probes)

    memorization_progress = trained_loss_before - trained_loss_after
    generalization_progress = sibling_loss_before - sibling_loss_after
    generalization_ratio = (
        generalization_progress / memorization_progress if memorization_progress > 0 else None
    )

    return {
        "category": get_category(topics, topic_name),
        "trained_loss_before": trained_loss_before, "trained_loss_after": trained_loss_after,
        "trained_acc_before": trained_acc_before, "trained_acc_after": trained_acc_after,
        "sibling_loss_before": sibling_loss_before, "sibling_loss_after": sibling_loss_after,
        "sibling_acc_before": sibling_acc_before, "sibling_acc_after": sibling_acc_after,
        "memorization_progress": memorization_progress,
        "generalization_progress": generalization_progress,
        "generalization_ratio": generalization_ratio,
    }


def run_noise_experiment(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu",
                          num_steps: int = 200, learning_rate: float = 1e-4,
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
        baseline_loss = (r["trained_loss_before"] + r["sibling_loss_before"]) / 2
        ratio_str = f"{r['generalization_ratio']:.3f}" if r["generalization_ratio"] is not None else "n/a"
        print(f"{name:<10} {r['category']:<10} {baseline_loss:>14.4f} {r['memorization_progress']:>13.4f} "
              f"{r['generalization_progress']:>15.4f} {ratio_str:>14}")

    return results


if __name__ == "__main__":
    run_noise_experiment()
