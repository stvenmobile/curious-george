import torch

from curious_george.deep_scoring import run_deep_scoring_pass
from curious_george.memory_store import MemoryStore, PipelineStatus, DeepScoreRecord
import curious_george.deep_scoring as deep_scoring_module


def _record(ratio_mean):
    return DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=2.0, generalization_mean=0.1,
        generalization_ratio_mean=ratio_mean, generalization_ratio_stdev=0.05,
    )


def test_run_deep_scoring_pass_promotes_positive_and_prunes_negative(monkeypatch):
    store = MemoryStore()
    store.add_or_update_item("good_topic", "content about something structured", {"m": torch.tensor([1.0, 0.0])})
    store.add_or_update_item("bad_topic", "content that turns out to be noise", {"m": torch.tensor([0.0, 1.0])})

    fake_records = {"good_topic": _record(0.25), "bad_topic": _record(-0.30)}

    def fake_deep_score_topic(model_name, device, content, **kwargs):
        # identify which topic by content, since deep_score_topic only sees content
        for topic, record in fake_records.items():
            if store.get_item(topic).content == content:
                return record
        raise AssertionError(f"unexpected content passed to deep_score_topic: {content!r}")

    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", fake_deep_score_topic)

    results = run_deep_scoring_pass(store, "fake-model", "cpu")

    assert results["good_topic"]["outcome"] == "active"
    assert results["bad_topic"]["outcome"] == "pruned"

    assert store.get_item("good_topic").status == PipelineStatus.ACTIVE
    assert store.get_item("good_topic").deep_score_history == [fake_records["good_topic"]]
    assert not store.has_topic("bad_topic"), "pruned items must be removed from the store entirely"
    assert len(store) == 1


def test_run_deep_scoring_pass_prunes_when_ratio_is_none(monkeypatch):
    """A None ratio means memorization_progress never even went
    positive - training didn't help the model predict its own content,
    a worse sign than a merely-negative-but-measurable ratio. Should be
    pruned, not kept on the technicality that None < 0.0 is False."""
    store = MemoryStore()
    store.add_or_update_item("unmeasurable_topic", "content", {"m": torch.tensor([1.0, 0.0])})

    unmeasurable_record = DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=-1.0, generalization_mean=0.0,
        generalization_ratio_mean=None, generalization_ratio_stdev=None,
    )
    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", lambda *a, **k: unmeasurable_record)

    results = run_deep_scoring_pass(store, "fake-model", "cpu")
    assert results["unmeasurable_topic"]["outcome"] == "pruned"
    assert not store.has_topic("unmeasurable_topic")


def test_run_deep_scoring_pass_only_scores_candidate_status_items_by_default(monkeypatch):
    store = MemoryStore()
    store.add_or_update_item("already_active", "content a", {"m": torch.tensor([1.0, 0.0])})
    store.set_status("already_active", PipelineStatus.ACTIVE)
    store.add_or_update_item("fresh_candidate", "content b", {"m": torch.tensor([0.0, 1.0])})

    calls = []

    def fake_deep_score_topic(model_name, device, content, **kwargs):
        calls.append(content)
        return _record(0.5)

    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", fake_deep_score_topic)

    results = run_deep_scoring_pass(store, "fake-model", "cpu")

    assert calls == ["content b"], "should only deep-score CANDIDATE-status items by default"
    assert "already_active" not in results
    assert "fresh_candidate" in results


def test_run_deep_scoring_pass_respects_explicit_topics_list_for_rescoring(monkeypatch):
    store = MemoryStore()
    store.add_or_update_item("already_active", "content a", {"m": torch.tensor([1.0, 0.0])})
    store.set_status("already_active", PipelineStatus.ACTIVE)

    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", lambda *a, **k: _record(0.5))

    results = run_deep_scoring_pass(store, "fake-model", "cpu", topics=["already_active"])
    assert "already_active" in results, "an explicit topics list should override the CANDIDATE-only default"
