"""
End-to-end harness for the warble/quaddle experiment.

Two entry points:

run_sanity_check - a PLUMBING sanity check, not a test of genuine
learning. The "study" step here is in-context only (study facts placed
directly in the prompt before each probe): if the facts are visible in
context at measurement time, loss improving mostly reflects ordinary
reading comprehension, not anything persisted in the model. What this
DOES meaningfully verify: MemoryStore population, real loss measurement
against a real model, persistence round-tripping, and that the quaddle
control stays completely isolated throughout - its own facts are never
touched, only warbles' context gets prepended to its probes, to check
whether irrelevant context generically helps prediction (it shouldn't,
for facts it doesn't answer).

run_lora_check - the genuine learning test. Both before and after
measurements are COLD - no warble content anywhere in the prompt, either
before or after fine-tuning. Any improvement can only come from the LoRA
adapter's own weights, not from generic context-priming.

Scoped down for run_sanity_check: only SmallModelEmbedder populates
MemoryStore, to avoid loading two separate copies of the same base model
(one via QwenHiddenStateEmbedder's own ResidualExtractor, one as the
model actually being measured).
"""

from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.curiosity_tools.memory_store import MemoryStore
from curious_george.validation_tools.fictional_entities import load_fictional_entities, get_study_facts, get_recall_probes
from curious_george.curiosity_tools.embeddings import SmallModelEmbedder, populate_item
from curious_george.curiosity_tools.loss_measurement import evaluate_recall_probes
from curious_george.curiosity_tools.lora_finetune import finetune_lora


def with_context_probes(probes: list, context_text: str) -> list:
    """Builds a modified probe list with study content prepended to each
    prompt - the in-context "study" step. Deliberately doesn't touch
    evaluate_recall_probes/compute_probe_loss at all; those stay exactly
    as already tested, this just constructs a different input for them."""
    return [{"prompt": f"{context_text} {p['prompt']}", "target": p["target"]} for p in probes]


def run_sanity_check(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu") -> dict:
    print(f"[Harness] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    entities = load_fictional_entities()
    warble_facts = get_study_facts(entities, "warbles")
    warble_context = " ".join(warble_facts)
    warble_probes = get_recall_probes(entities, "warbles")
    quaddle_probes = get_recall_probes(entities, "quaddles")  # control - facts NEVER touched

    print("[Harness] Populating MemoryStore with warbles (small_model embeddings only, this pass)...")
    store = MemoryStore()
    small_embedder = SmallModelEmbedder()
    populate_item(store, "warbles", warble_context, {small_embedder.METHOD_NAME: small_embedder})

    print("[Harness] Measuring BASELINE loss (before study) on warbles and quaddles...")
    warble_loss_before, warble_acc_before = evaluate_recall_probes(model, tokenizer, device, warble_probes)
    quaddle_loss_before, quaddle_acc_before = evaluate_recall_probes(model, tokenizer, device, quaddle_probes)
    store.record_loss("warbles", warble_loss_before)

    print("[Harness] 'Studying' warbles (in-context - see module docstring for why this is a "
          "sanity check, not a learning test)...")
    warble_probes_with_context = with_context_probes(warble_probes, warble_context)
    # Quaddle probes get WARBLE context prepended, never their own facts -
    # tests whether irrelevant context generically helps prediction.
    quaddle_probes_with_warble_context = with_context_probes(quaddle_probes, warble_context)

    print("[Harness] Measuring loss AFTER study...")
    warble_loss_after, warble_acc_after = evaluate_recall_probes(model, tokenizer, device, warble_probes_with_context)
    quaddle_loss_after, quaddle_acc_after = evaluate_recall_probes(model, tokenizer, device, quaddle_probes_with_warble_context)
    store.record_loss("warbles", warble_loss_after)

    progress = store.learning_progress("warbles")

    print(f"\n[Harness] warbles   loss: {warble_loss_before:.4f} -> {warble_loss_after:.4f}  "
          f"(delta {progress:+.4f})   accuracy: {warble_acc_before:.3f} -> {warble_acc_after:.3f}")
    print(f"[Harness] quaddles  loss: {quaddle_loss_before:.4f} -> {quaddle_loss_after:.4f}  "
          f"(delta {quaddle_loss_before - quaddle_loss_after:+.4f})   "
          f"accuracy: {quaddle_acc_before:.3f} -> {quaddle_acc_after:.3f}")

    print("\n[Harness] Checking persistence round-trip (save/reload)...")
    import tempfile
    scratch = Path(tempfile.mkdtemp()) / "sanity_check_memory"
    store.save(scratch)
    reloaded = MemoryStore.load(scratch)
    assert reloaded.learning_progress("warbles") == progress, "learning_progress must survive save/load"
    print("[Harness] Persistence check OK - learning_progress survives a save/load round-trip.")

    return {
        "warble_loss_before": warble_loss_before, "warble_loss_after": warble_loss_after,
        "quaddle_loss_before": quaddle_loss_before, "quaddle_loss_after": quaddle_loss_after,
        "warble_learning_progress": progress,
    }


def run_lora_check(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu",
                    num_steps: int = 200, learning_rate: float = 1e-4) -> dict:
    """The genuine learning test, unlike run_sanity_check above: both
    before and after measurements are COLD - no warble content anywhere
    in the prompt at measurement time, either before or after fine-tuning.
    Any improvement can only come from the LoRA adapter's own weights,
    not from generic context-priming (the confound the in-context version
    demonstrated with real numbers).

    Quaddle probes stay completely cold too, with NO context prepended at
    all here (unlike run_sanity_check's warble-context-on-quaddle-probes
    check) - the LoRA adapter was never trained on anything resembling
    quaddle content, so there's no context to even construct; the
    question is simply whether fine-tuning on warbles alone leaked into
    quaddle predictions through the adapter's own weights."""
    print(f"[LoRA Check] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()

    entities = load_fictional_entities()
    warble_facts = get_study_facts(entities, "warbles")
    warble_probes = get_recall_probes(entities, "warbles")
    quaddle_probes = get_recall_probes(entities, "quaddles")  # control - facts NEVER touched

    print("[LoRA Check] Measuring COLD baseline loss (before LoRA) on warbles and quaddles...")
    warble_loss_before, warble_acc_before = evaluate_recall_probes(base_model, tokenizer, device, warble_probes)
    quaddle_loss_before, quaddle_acc_before = evaluate_recall_probes(base_model, tokenizer, device, quaddle_probes)

    print(f"[LoRA Check] Fine-tuning LoRA adapter on {len(warble_facts)} warble facts, {num_steps} steps...")
    tuned_model = finetune_lora(
        base_model, tokenizer, device, warble_facts, num_steps=num_steps, learning_rate=learning_rate,
    )

    print("[LoRA Check] Measuring COLD loss AFTER LoRA fine-tuning (no warble content in the prompt)...")
    warble_loss_after, warble_acc_after = evaluate_recall_probes(tuned_model, tokenizer, device, warble_probes)
    quaddle_loss_after, quaddle_acc_after = evaluate_recall_probes(tuned_model, tokenizer, device, quaddle_probes)

    warble_progress = warble_loss_before - warble_loss_after
    quaddle_progress = quaddle_loss_before - quaddle_loss_after

    print(f"\n[LoRA Check] warbles   loss: {warble_loss_before:.4f} -> {warble_loss_after:.4f}  "
          f"(delta {warble_progress:+.4f})   accuracy: {warble_acc_before:.3f} -> {warble_acc_after:.3f}")
    print(f"[LoRA Check] quaddles  loss: {quaddle_loss_before:.4f} -> {quaddle_loss_after:.4f}  "
          f"(delta {quaddle_progress:+.4f})   accuracy: {quaddle_acc_before:.3f} -> {quaddle_acc_after:.3f}")

    if quaddle_progress > 0:
        leak_fraction = quaddle_progress / warble_progress if warble_progress else float("inf")
        print(f"\n[LoRA Check] Quaddle-to-warble improvement ratio: {leak_fraction:.3f} "
              f"(in-context sanity check's confound was ~0.436 - lower here means LoRA is more "
              f"fact-specific, not just repeating the same leak in a different form)")

    return {
        "warble_loss_before": warble_loss_before, "warble_loss_after": warble_loss_after,
        "quaddle_loss_before": quaddle_loss_before, "quaddle_loss_after": quaddle_loss_after,
        "warble_learning_progress": warble_progress, "quaddle_learning_progress": quaddle_progress,
    }


if __name__ == "__main__":
    run_sanity_check()
