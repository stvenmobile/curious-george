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

import statistics
from typing import Dict, Iterable, List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.curiosity_topics import (
    load_curiosity_topics, get_train_facts, get_trained_probes, get_sibling_probes, get_category,
)
from curious_george.loss_measurement import evaluate_recall_probes
from curious_george.lora_finetune import finetune_lora


def run_topic_trial(model_name: str, device: str, topic_name: str, topics: dict = None,
                     num_steps: int = 200, learning_rate: float = 1e-4, seed: int = 1234) -> dict:
    """Runs one topic's LoRA study step and returns both the
    memorization and generalization (sibling-topic) deltas. `topics`
    can be passed in (already loaded) so run_noise_experiment doesn't
    reload the JSON file once per topic; loads it itself if omitted, so
    this still works standalone.

    `seed` controls finetune_lora's fact-sampling order - a single run
    at a fixed seed is one sample from a noisy process, not a
    guaranteed representative one. run_repeated_trials exists precisely
    to vary this across several independent runs rather than trust one."""
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
        base_model, tokenizer, device, train_facts, num_steps=num_steps, learning_rate=learning_rate,
        seed=seed, verbose=False,
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
    """Runs run_topic_trial ONCE for every topic and prints a comparison
    table. A single run per topic is one sample from a noisy process -
    see run_noise_experiment_repeated for the version that actually
    measures run-to-run variance instead of trusting one draw."""
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


def summarize_trials(trials: List[dict]) -> dict:
    """Mean/stdev across a list of run_topic_trial results for one
    topic. stdev needs at least 2 points to be defined; returns 0.0 for
    a single trial rather than raising, so this stays usable while
    iterating on trial count. generalization_ratio is excluded from the
    average on any trial where it was None (memorization_progress <= 0),
    rather than treating that trial as a 0 - a None ratio means "not
    meaningful," not "no transfer"."""
    memorization = [t["memorization_progress"] for t in trials]
    generalization = [t["generalization_progress"] for t in trials]
    ratios = [t["generalization_ratio"] for t in trials if t["generalization_ratio"] is not None]

    def mean_and_stdev(values):
        if not values:
            return None, None
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, stdev

    memorization_mean, memorization_stdev = mean_and_stdev(memorization)
    generalization_mean, generalization_stdev = mean_and_stdev(generalization)
    ratio_mean, ratio_stdev = mean_and_stdev(ratios)

    return {
        "n": len(trials),
        "memorization_mean": memorization_mean, "memorization_stdev": memorization_stdev,
        "generalization_mean": generalization_mean, "generalization_stdev": generalization_stdev,
        "ratio_mean": ratio_mean, "ratio_stdev": ratio_stdev, "ratio_n": len(ratios),
    }


def run_repeated_trials(model_name: str, device: str, topic_name: str, topics: dict = None,
                         num_trials: int = 5, num_steps: int = 200, learning_rate: float = 1e-4,
                         base_seed: int = 1) -> List[dict]:
    """Runs run_topic_trial num_trials times for one topic, with a
    different LoRA training seed each time (base_seed, base_seed+1, ...)
    so the resulting spread reflects genuine run-to-run variance rather
    than one possibly-unrepresentative measurement."""
    if topics is None:
        topics = load_curiosity_topics()

    trials = []
    for i in range(num_trials):
        seed = base_seed + i
        print(f"[Curiosity] Topic '{topic_name}', trial {i + 1}/{num_trials} (seed={seed})...")
        result = run_topic_trial(model_name, device, topic_name, topics=topics,
                                  num_steps=num_steps, learning_rate=learning_rate, seed=seed)
        result["seed"] = seed
        trials.append(result)
        ratio_str = f"{result['generalization_ratio']:.3f}" if result["generalization_ratio"] is not None else "n/a"
        print(f"  memorization={result['memorization_progress']:.4f}  "
              f"generalization={result['generalization_progress']:.4f}  ratio={ratio_str}")

    return trials


def run_noise_experiment_repeated(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu",
                                   num_trials: int = 5, num_steps: int = 200, learning_rate: float = 1e-4,
                                   topic_names: Iterable[str] = ("known", "moderate", "noise"),
                                   base_seed: int = 1) -> Dict[str, dict]:
    """Runs run_repeated_trials for every topic and prints per-trial
    results plus a mean +/- stdev summary table. This is the version
    whose numbers are worth recording as a real finding - a single run
    per topic (run_noise_experiment) is one draw from a noisy process."""
    topics = load_curiosity_topics()
    all_trials = {}
    summaries = {}

    for name in topic_names:
        print(f"\n=== {name} ({get_category(topics, name)}) ===")
        trials = run_repeated_trials(model_name, device, name, topics=topics, num_trials=num_trials,
                                      num_steps=num_steps, learning_rate=learning_rate, base_seed=base_seed)
        all_trials[name] = trials
        summaries[name] = summarize_trials(trials)

    print(f"\n{'topic':<10} {'n':>3} {'mem mean':>10} {'mem stdev':>10} "
          f"{'gen mean':>10} {'gen stdev':>10} {'ratio mean':>11} {'ratio stdev':>12}")
    for name, s in summaries.items():
        ratio_mean_str = f"{s['ratio_mean']:.3f}" if s["ratio_mean"] is not None else "n/a"
        ratio_stdev_str = f"{s['ratio_stdev']:.3f}" if s["ratio_stdev"] is not None else "n/a"
        print(f"{name:<10} {s['n']:>3} {s['memorization_mean']:>10.4f} {s['memorization_stdev']:>10.4f} "
              f"{s['generalization_mean']:>10.4f} {s['generalization_stdev']:>10.4f} "
              f"{ratio_mean_str:>11} {ratio_stdev_str:>12}")

    return {"trials": all_trials, "summaries": summaries}


if __name__ == "__main__":
    run_noise_experiment()
