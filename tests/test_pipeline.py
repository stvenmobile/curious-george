import torch

from curious_george.curiosity_tools.pipeline import add_candidate, add_candidates
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus


class FakeEmbedder:
    METHOD_NAME = "fake_method"

    def __init__(self, vectors):
        self.vectors = vectors  # content -> tensor

    def embed(self, text):
        return self.vectors[text]


class FakeModelWithLoss:
    """Returns a caller-controlled loss per content string, so tests can
    place a candidate at a specific mastery tier without needing a real
    model."""
    def __init__(self, losses):
        self.losses = losses  # token-id-tuple -> loss value

    def __call__(self, input_ids, attention_mask, labels):
        key = tuple(input_ids[0].tolist())

        class FakeOutputs:
            loss = torch.tensor(self.losses[key])

        return FakeOutputs()


class FakeTokenizer:
    def __init__(self, content_to_ids):
        self.content_to_ids = content_to_ids

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = self.content_to_ids[text]
        return _FakeBatchEncoding(input_ids=torch.tensor([ids], dtype=torch.long))


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


def test_add_candidate_records_embedding_baseline_loss_and_mastery():
    content = "Some novel content."
    vector = torch.tensor([1.0, 0.0])
    tokenizer = FakeTokenizer({content: [1, 2, 3]})
    model = FakeModelWithLoss({(1, 2, 3): 3.0})  # -> "novice" per classify_mastery
    embedder = FakeEmbedder({content: vector})

    store = MemoryStore()
    result = add_candidate(store, model, tokenizer, "cpu", embedder, "topic_a", content)

    assert result["topic"] == "topic_a"
    assert result["baseline_loss"] == 3.0
    assert result["mastery"] == "novice"
    assert result["resonance"] is None, "no interest_embeddings passed - resonance should be skipped, not guessed at"
    assert result["status"] == PipelineStatus.CANDIDATE

    item = store.get_item("topic_a")
    assert item.content == content
    assert item.loss_history == [item.loss_history[-1]], "exactly one baseline reading recorded"
    assert torch.equal(store.embedding_matrices["fake_method"][0], vector)


def test_add_candidate_computes_resonance_when_interests_are_given():
    content = "Content about agents."
    vector = torch.tensor([1.0, 0.0])
    tokenizer = FakeTokenizer({content: [5]})
    model = FakeModelWithLoss({(5,): 5.0})
    embedder = FakeEmbedder({content: vector})
    interests = {"agentic behavior": torch.tensor([1.0, 0.0]), "free will": torch.tensor([0.0, 1.0])}

    store = MemoryStore()
    result = add_candidate(store, model, tokenizer, "cpu", embedder, "topic_b", content,
                            interest_embeddings=interests)

    assert abs(result["resonance"] - 1.0) < 1e-6, "should match the closest declared interest exactly"


def test_add_candidates_batches_over_a_list():
    contents = {"first content": [1], "second content": [2]}
    vectors = {"first content": torch.tensor([1.0, 0.0]), "second content": torch.tensor([0.0, 1.0])}
    tokenizer = FakeTokenizer(contents)
    model = FakeModelWithLoss({(1,): 8.0, (2,): 0.5})  # newbie, master
    embedder = FakeEmbedder(vectors)

    store = MemoryStore()
    candidates = [
        {"topic": "topic_1", "content": "first content"},
        {"topic": "topic_2", "content": "second content"},
    ]
    results = add_candidates(store, model, tokenizer, "cpu", embedder, candidates)

    assert len(store) == 2
    assert [r["topic"] for r in results] == ["topic_1", "topic_2"]
    assert results[0]["mastery"] == "newbie"
    assert results[1]["mastery"] == "master"
