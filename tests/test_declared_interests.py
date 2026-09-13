import torch

from curious_george.declared_interests import load_declared_interests, embed_declared_interests


def test_real_content_loads_with_expected_shape():
    interests = load_declared_interests()
    assert len(interests) == 5
    names = {entry["name"] for entry in interests}
    assert names == {
        "human curiosity", "agentic behavior", "goal prioritization",
        "memory as a predictor", "free will",
    }
    for entry in interests:
        assert set(entry.keys()) == {"name", "description"}
        assert len(entry["description"]) > 20, f"description for {entry['name']!r} looks too thin to embed meaningfully"


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return torch.tensor([float(len(text)), 0.0])


def test_embed_declared_interests_embeds_the_description_not_the_name():
    interests = [
        {"name": "short", "description": "a longer description text"},
        {"name": "other", "description": "another description"},
    ]
    embedder = FakeEmbedder()
    embeddings = embed_declared_interests(interests, embedder)

    assert set(embeddings.keys()) == {"short", "other"}
    assert embedder.calls == ["a longer description text", "another description"], (
        "should embed each interest's description, not its bare name"
    )
    assert torch.equal(embeddings["short"], torch.tensor([float(len("a longer description text")), 0.0]))
