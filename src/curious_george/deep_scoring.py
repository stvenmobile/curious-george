"""
Deep scoring: the expensive transferability check for real pipeline
candidates, plus the prune/promote pass over MemoryStore.

See canary_topics.py for why this measures generalization-quality
against a fixed canary pool rather than Phase 1's domain-matched
sibling design, and obsidian/Journals/2026-09-13.md for why this is
averaged across 5 trials rather than trusted from a single one (Phase 1
measured real run-to-run variance directly - moderate's ratio ranged
0.126-0.322 across 5 seeds, same topic, same everything but the
training seed).

"Memorization" here is measured via measure_content_loss on the
candidate's own content, not evaluate_recall_probes - real candidates
don't arrive with hand-built recall_probes the way Phase 0/1's fixtures
did; they're just whatever content a caller (or, later, Piper) proposed.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.canary_topics import load_canary_topics, all_canary_probes
from curious_george.curiosity import summarize_trials
from curious_george.loss_measurement import measure_content_loss, evaluate_recall_probes
from curious_george.lora_finetune import finetune_lora
from curious_george.memory_store import MemoryStore, DeepScoreRecord, PipelineStatus


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_score_trial(model_name: str, device: str, content: str, canary_probes: List[Dict[str, str]],
                      num_steps: int = 200, learning_rate: float = 1e-4, seed: int = 1234) -> dict:
    """One deep-scoring trial for a single piece of real content.
    Mirrors curiosity.run_topic_trial's before/study/after pattern, with
    the two substitutions described in this module's docstring."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()

    content_loss_before = measure_content_loss(base_model, tokenizer, device, content)
    canary_loss_before, _ = evaluate_recall_probes(base_model, tokenizer, device, canary_probes)

    tuned_model = finetune_lora(
        base_model, tokenizer, device, [content], num_steps=num_steps, learning_rate=learning_rate,
        seed=seed, verbose=False,
    )

    content_loss_after = measure_content_loss(tuned_model, tokenizer, device, content)
    canary_loss_after, _ = evaluate_recall_probes(tuned_model, tokenizer, device, canary_probes)

    memorization_progress = content_loss_before - content_loss_after
    generalization_progress = canary_loss_before - canary_loss_after
    generalization_ratio = (
        generalization_progress / memorization_progress if memorization_progress > 0 else None
    )

    return {
        "content_loss_before": content_loss_before, "content_loss_after": content_loss_after,
        "canary_loss_before": canary_loss_before, "canary_loss_after": canary_loss_after,
        "memorization_progress": memorization_progress,
        "generalization_progress": generalization_progress,
        "generalization_ratio": generalization_ratio,
    }


def deep_score_topic(model_name: str, device: str, content: str, num_trials: int = 5,
                      num_steps: int = 200, learning_rate: float = 1e-4, base_seed: int = 1,
                      canary_probes: Optional[List[Dict[str, str]]] = None) -> DeepScoreRecord:
    """Runs deep_score_trial num_trials times, a different seed each
    time, and returns one DeepScoreRecord summarizing the average - the
    unit that actually gets stored via MemoryStore.record_deep_score.
    A single trial isn't trustworthy enough to prune or promote on; see
    the module docstring."""
    if canary_probes is None:
        canary_probes = all_canary_probes(load_canary_topics())

    trials = [
        deep_score_trial(model_name, device, content, canary_probes,
                          num_steps=num_steps, learning_rate=learning_rate, seed=base_seed + i)
        for i in range(num_trials)
    ]
    summary = summarize_trials(trials)

    return DeepScoreRecord(
        timestamp=_now_iso(),
        n_trials=summary["n"],
        memorization_mean=summary["memorization_mean"],
        generalization_mean=summary["generalization_mean"],
        generalization_ratio_mean=summary["ratio_mean"],
        generalization_ratio_stdev=summary["ratio_stdev"],
    )


def run_deep_scoring_pass(store: MemoryStore, model_name: str, device: str,
                           keep_top_n: Optional[int] = None,
                           num_trials: int = 5, num_steps: int = 200, learning_rate: float = 1e-4,
                           topics: Optional[List[str]] = None) -> Dict[str, dict]:
    """Deep-scores every requested topic (defaults to every CANDIDATE-
    status item in the store - the ones never yet deep-scored), then
    ranks by mean generalization ratio and keeps only the top
    `keep_top_n` - everything else is pruned.

    This replaced an earlier absolute-threshold version (prune anything
    below 0.0) after a real batch of non-fixture candidates showed why
    that doesn't work: canary-based scoring measures transfer against a
    fixed, UNRELATED sibling rather than a domain-matched one (see
    canary_topics.py), and real, isolated content without a deliberately
    matched partner tends to show modest negative transfer almost across
    the board - the same thing Phase 1's own "known" category showed.
    An absolute zero cutoff pruned an entire real batch even though the
    ranking within it was genuinely informative (one candidate was an
    order of magnitude less negative than the other two). Relative
    ranking surfaces that signal; an absolute cutoff throws it away.

    Two things ALWAYS get pruned regardless of rank: an unmeasurable
    ratio (memorization_progress never even went positive - a red flag
    about the trial itself, not just "worse than its peers"), and,
    naturally, anything that doesn't make the top_n cut. Passing
    keep_top_n=None keeps every measurable item (only the unmeasurable
    ones get pruned) - useful when there isn't yet enough real data to
    pick a defensible N.

    Deliberately does not archive anything here - archiving (mastered,
    no new sources currently available) is a judgment about mastery and
    external source availability, not something a deep-scoring pass can
    decide on its own. See obsidian/Journals/2026-09-13.md.

    Pass an explicit `topics` list to rescore specific items instead
    (e.g. after further real study on them, per the rescoring policy -
    "new research conducted on an existing item" is exactly the case
    that calls for this, not a fresh scan of every CANDIDATE)."""
    if topics is None:
        topics = [item.topic for item in store.items if item.status == PipelineStatus.CANDIDATE]

    canary_probes = all_canary_probes(load_canary_topics())
    records: Dict[str, DeepScoreRecord] = {}

    for topic in topics:
        item = store.get_item(topic)
        record = deep_score_topic(model_name, device, item.content, num_trials=num_trials,
                                   num_steps=num_steps, learning_rate=learning_rate,
                                   canary_probes=canary_probes)
        store.record_deep_score(topic, record)
        records[topic] = record

    measurable = {t: r for t, r in records.items() if r.generalization_ratio_mean is not None}
    ranked = sorted(measurable.items(), key=lambda kv: kv[1].generalization_ratio_mean, reverse=True)
    kept = {t for t, _ in (ranked if keep_top_n is None else ranked[:keep_top_n])}

    results = {}
    for topic, record in records.items():
        if topic in kept:
            store.set_status(topic, PipelineStatus.ACTIVE)
            results[topic] = {"outcome": "active", "record": record}
        else:
            store.remove_item(topic)
            results[topic] = {"outcome": "pruned", "record": record}

    return results
