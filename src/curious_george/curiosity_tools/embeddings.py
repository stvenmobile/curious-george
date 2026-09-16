"""
Embedding methods, run side by side.

Two candidates, compared empirically rather than settled on paper. A
dedicated small sentence-embedding model is purpose-built for retrieval
(trained via contrastive learning so cosine similarity directly reflects
semantic relevance). Reusing Qwen's own hidden states (via
ResidualExtractor, ported from piper_assistant's cross-model reasoning-
geometry work) is a repurposed by-product of a next-token-prediction
objective instead - real, measured structure exists there, but with
well-earned low confidence about any single layer.

Both expose the same embed(text) -> torch.Tensor interface, so code that
populates MemoryStore can treat them uniformly without caring which one
it's using.
"""

from typing import Dict

import torch


class SmallModelEmbedder:
    """Wraps a small sentence-embedding model. Default is
    all-MiniLM-L6-v2 - tiny (~90MB), CPU-friendly, and the most
    widely-used default for exactly this job."""

    METHOD_NAME = "small_model"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> torch.Tensor:
        vector = self.model.encode(text, convert_to_tensor=True)
        return vector.to(torch.float32).cpu()


class QwenHiddenStateEmbedder:
    """Repurposes Qwen's own hidden state at a given layer as an
    embedding, via ResidualExtractor. extract_activations() L2-normalizes
    what it returns; harmless here, since cosine similarity only cares
    about direction, not magnitude.

    layer=18 matches the layer the reasoning-geometry cross-model work
    used (~75% depth) - reused here as a reasonable starting point, not
    because it's been validated as the best layer for retrieval
    specifically. That's a genuinely open, empirically-answerable
    question, not assumed settled by reusing the number."""

    METHOD_NAME = "qwen_hidden"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", layer: int = 18, device: str = "cuda"):
        from curious_george.curiosity_tools.extractor import ResidualExtractor
        self.extractor = ResidualExtractor(model_name_or_path=model_name, device=device)
        self.layer = layer

    def embed(self, text: str) -> torch.Tensor:
        activations = self.extractor.extract_activations(prompt=text, target_layers=[self.layer])
        return activations[self.layer].to(torch.float32)


def populate_item(store, topic: str, content: str, embedders: Dict[str, object]):
    """Computes an embedding from every embedder in `embedders` (keyed by
    METHOD_NAME) and hands them all to MemoryStore.add_or_update_item in
    one call, so a caller adding real content never has to manually keep
    two embedding dicts and a store call in sync."""
    embeddings = {name: embedder.embed(content) for name, embedder in embedders.items()}
    return store.add_or_update_item(topic, content, embeddings)
