"""
The interest pipeline: cheap entry-scoring for new candidate topics.

Per obsidian/Journals/2026-09-13.md's two-tier cost model: getting a
candidate into MemoryStore only needs a cheap measurement - one baseline
loss reading (measure_content_loss, no training, no hand-authored
probes) plus an embedding - not a full LoRA trial. That's affordable for
a whole batch of suggestions at once. Deep scoring (transferability, the
expensive signal that actually distinguishes learnable structure from
noise - see notebooks/curious_george.ipynb's Phase 1 result) is reserved
for whatever's actually being considered for idle-time research, not run
here.
"""

from typing import Dict, List, Optional

import torch

from curious_george.loss_measurement import measure_content_loss
from curious_george.memory_store import MemoryStore


def add_candidate(store: MemoryStore, model, tokenizer, device: str, embedder,
                   topic: str, content: str,
                   interest_embeddings: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    """Adds (or re-studies, via MemoryStore's existing semantics) one
    candidate topic with only the cheap measurements: a baseline loss
    reading and an embedding. New items land at PipelineStatus.CANDIDATE
    by default (see MemoryItem) - this function doesn't promote, prune,
    or archive anything, that's deep-scoring's job.

    Returns a small summary rather than the raw MemoryItem, since a
    caller adding a batch of suggestions mostly just wants to know where
    each one landed (mastery, resonance) without digging through the
    store afterward."""
    embedding = embedder.embed(content)
    store.add_or_update_item(topic, content, {embedder.METHOD_NAME: embedding})

    baseline_loss = measure_content_loss(model, tokenizer, device, content)
    store.record_loss(topic, baseline_loss)

    resonance = None
    if interest_embeddings:
        resonance = store.resonance(topic, embedder.METHOD_NAME, interest_embeddings)

    return {
        "topic": topic,
        "baseline_loss": baseline_loss,
        "mastery": store.mastery_level(topic),
        "resonance": resonance,
        "status": store.get_item(topic).status,
    }


def add_candidates(store: MemoryStore, model, tokenizer, device: str, embedder,
                    candidates: List[Dict[str, str]],
                    interest_embeddings: Optional[Dict[str, torch.Tensor]] = None) -> List[dict]:
    """add_candidate over a whole batch - candidates is a list of
    {"topic": ..., "content": ...} dicts, e.g. a batch of suggestions or
    pseudo-experiential content (see the journal for that framing)."""
    return [
        add_candidate(store, model, tokenizer, device, embedder,
                      candidate["topic"], candidate["content"], interest_embeddings)
        for candidate in candidates
    ]
