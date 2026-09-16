"""
Real study: committing actual LoRA study time to a topic, as opposed to
deep-scoring's disposable measurement trials.

Only runs on ACTIVE-status items (survived deep-scoring) - real study
time shouldn't go to a candidate that was never confirmed to have
transferable structure, let alone something already archived. This is
the "committed investment" tier the pipeline has been staged toward:
cheap entry-scoring -> expensive deep-scoring -> real study, each tier
spending more only on what earned its way past the one before it.

IMPORTANT LIMITATION, stated plainly rather than glossed over: like
every other LoRA trial in this project, this reloads a fresh base model
and discards the trained adapter afterward - nothing persists between
calls. That means calling study_topic twice on the same topic will NOT
show rising cumulative mastery; the second session starts from the same
frozen base model as the first. What this DOES measure, honestly and
repeatably: "if committed study happened right now, how much would loss
improve" - a real, useful signal, just not the same thing as mastery
actually accumulating over calendar time. True accumulation requires
resolving whether real study should persist into a shared, evolving
model - a genuinely open question, deliberately deferred (see
obsidian/Journals/2026-09-13.md), not solved here.

Re-deep-scoring after a real study session is a separate, explicit
step - pass topics=[topic] to deep_scoring.run_deep_scoring_pass, which
already supports exactly this case (built during the rescoring-policy
design, before this module existed).
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

from curious_george.curiosity_tools.loss_measurement import measure_content_loss
from curious_george.curiosity_tools.lora_finetune import finetune_lora
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus


def study_topic(store: MemoryStore, model_name: str, device: str, topic: str,
                 num_steps: int = 200, learning_rate: float = 1e-4, seed: int = 1234) -> dict:
    """One real study session. Measures cold loss on the topic's own
    content before and after a full LoRA study step, records the
    post-study loss into loss_history (the ongoing trace
    learning_progress/mastery_level read from), and updates
    last_studied. Raises if the topic doesn't exist or isn't ACTIVE."""
    item = store.get_item(topic)
    if item is None:
        raise KeyError(f"cannot study {topic!r} - it has never been added to the store")
    if item.status != PipelineStatus.ACTIVE:
        raise ValueError(
            f"{topic!r} has status={item.status.value!r}, not 'active' - only topics that have "
            f"survived deep-scoring should get real study time committed to them"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()

    loss_before = measure_content_loss(base_model, tokenizer, device, item.content)

    tuned_model = finetune_lora(
        base_model, tokenizer, device, [item.content],
        num_steps=num_steps, learning_rate=learning_rate, seed=seed, verbose=False,
    )

    loss_after = measure_content_loss(tuned_model, tokenizer, device, item.content)

    store.record_loss(topic, loss_after)
    store.mark_studied(topic)

    return {
        "topic": topic,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "progress_this_session": loss_before - loss_after,
        "mastery": store.mastery_level(topic),
    }
