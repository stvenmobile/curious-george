import torch

from curious_george.curiosity_tools.embeddings import SmallModelEmbedder, QwenHiddenStateEmbedder, populate_item
from curious_george.curiosity_tools.memory_store import MemoryStore


def test_small_model_embedder_shape_dtype_and_semantic_sanity():
    """Runs against the REAL model (small/CPU-friendly enough to run for
    real here, unlike Qwen) - shape, dtype, and a real semantic-sanity
    check: two similarly-structured animal-fact sentences should score
    higher cosine similarity than either does against an unrelated
    sentence."""
    embedder = SmallModelEmbedder()
    assert embedder.METHOD_NAME == "small_model"

    v_warble = embedder.embed("Warbles are green and yellow mammals.")
    v_quaddle = embedder.embed("Quaddles are blue and gray reptiles.")
    v_unrelated = embedder.embed("The quick brown fox jumps over the lazy dog.")

    assert v_warble.shape == (384,), f"expected 384-dim MiniLM output, got {v_warble.shape}"
    assert v_warble.dtype == torch.float32

    cos = torch.nn.functional.cosine_similarity
    sim_structural = cos(v_warble.unsqueeze(0), v_quaddle.unsqueeze(0)).item()
    sim_unrelated = cos(v_warble.unsqueeze(0), v_unrelated.unsqueeze(0)).item()
    assert sim_structural > sim_unrelated, (
        f"expected two similarly-structured animal-fact sentences ({sim_structural:.3f}) to score higher than "
        f"an unrelated sentence ({sim_unrelated:.3f}) - the embedding space should pick up real structure"
    )


def test_qwen_hidden_state_embedder_calls_extractor_correctly(monkeypatch):
    """Synthetic fake for the ResidualExtractor dependency - real
    verification of ResidualExtractor itself lives wherever Qwen is
    already cached; this just checks the thin wrapper's own logic."""
    class FakeResidualExtractor:
        def __init__(self, model_name_or_path, device):
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.last_prompt = None
            self.last_target_layers = None

        def extract_activations(self, prompt, target_layers):
            self.last_prompt = prompt
            self.last_target_layers = target_layers
            (layer,) = target_layers
            return {layer: torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)}  # half precision, like the real one

    import curious_george.curiosity_tools.extractor as extractor_module
    monkeypatch.setattr(extractor_module, "ResidualExtractor", FakeResidualExtractor)

    qwen_embedder = QwenHiddenStateEmbedder(model_name="fake-model", layer=18, device="cpu")
    assert qwen_embedder.METHOD_NAME == "qwen_hidden"
    result = qwen_embedder.embed("Warbles are green and yellow mammals.")

    assert qwen_embedder.extractor.last_prompt == "Warbles are green and yellow mammals."
    assert qwen_embedder.extractor.last_target_layers == [18]
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))
    assert result.dtype == torch.float32, "embed() should cast the extractor's fp16 output to float32"


def test_populate_item_orchestrates_multiple_embedders():
    class FakeEmbedderA:
        METHOD_NAME = "method_a"

        def embed(self, text):
            return torch.tensor([1.0, 0.0])

    class FakeEmbedderB:
        METHOD_NAME = "method_b"

        def embed(self, text):
            return torch.tensor([0.0, 1.0, 0.0])

    store = MemoryStore()
    item = populate_item(store, "warbles", "Warbles are green and yellow mammals.", {
        "method_a": FakeEmbedderA(), "method_b": FakeEmbedderB(),
    })

    assert item.topic == "warbles"
    assert item.content == "Warbles are green and yellow mammals."
    assert torch.equal(store.embedding_matrices["method_a"][0], torch.tensor([1.0, 0.0]))
    assert torch.equal(store.embedding_matrices["method_b"][0], torch.tensor([0.0, 1.0, 0.0]))
