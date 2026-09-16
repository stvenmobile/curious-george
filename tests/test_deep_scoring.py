import torch

from curious_george.curiosity_tools.deep_scoring import run_deep_scoring_pass
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus, DeepScoreRecord
import curious_george.curiosity_tools.deep_scoring as deep_scoring_module


def _record(ratio_mean):
    return DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=2.0, generalization_mean=0.1,
        generalization_ratio_mean=ratio_mean, generalization_ratio_stdev=0.05,
    )


def _fake_scorer_by_content(store, fake_records: dict):
    """fake_records maps topic -> DeepScoreRecord; deep_score_topic only
    ever sees content, so this looks up which topic a piece of content
    belongs to."""
    def fake_deep_score_topic(model_name, device, content, **kwargs):
        for topic, record in fake_records.items():
            if store.get_item(topic).content == content:
                return record
        raise AssertionError(f"unexpected content passed to deep_score_topic: {content!r}")
    return fake_deep_score_topic


def test_run_deep_scoring_pass_keeps_only_top_n_by_rank(monkeypatch):
    store = MemoryStore()
    store.add_or_update_item("good_topic", "content about something structured", {"m": torch.tensor([1.0, 0.0])})
    store.add_or_update_item("bad_topic", "content that turns out to be noise", {"m": torch.tensor([0.0, 1.0])})

    fake_records = {"good_topic": _record(0.25), "bad_topic": _record(-0.30)}
    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", _fake_scorer_by_content(store, fake_records))

    results = run_deep_scoring_pass(store, "fake-model", "cpu", keep_top_n=1)

    assert results["good_topic"]["outcome"] == "active"
    assert results["bad_topic"]["outcome"] == "pruned"

    assert store.get_item("good_topic").status == PipelineStatus.ACTIVE
    assert store.get_item("good_topic").deep_score_history == [fake_records["good_topic"]]
    assert not store.has_topic("bad_topic"), "pruned items must be removed from the store entirely"
    assert len(store) == 1


def test_run_deep_scoring_pass_keeps_top_n_across_more_than_two_candidates(monkeypatch):
    store = MemoryStore()
    store.add_or_update_item("best", "content best", {"m": torch.tensor([1.0, 0.0])})
    store.add_or_update_item("middle", "content middle", {"m": torch.tensor([0.0, 1.0])})
    store.add_or_update_item("worst", "content worst", {"m": torch.tensor([1.0, 1.0])})

    fake_records = {"best": _record(0.30), "middle": _record(-0.02), "worst": _record(-0.10)}
    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", _fake_scorer_by_content(store, fake_records))

    results = run_deep_scoring_pass(store, "fake-model", "cpu", keep_top_n=2)

    assert results["best"]["outcome"] == "active"
    assert results["middle"]["outcome"] == "active"
    assert results["worst"]["outcome"] == "pruned", "lowest-ranked of the three should be the one cut at N=2"


def test_run_deep_scoring_pass_keeps_every_measurable_item_when_keep_top_n_is_none(monkeypatch):
    """The default (None) applies no relative-rank pruning at all - only
    the hard unmeasurable rule below applies. Useful when there isn't
    yet enough real data to pick a defensible N; even a negative-ratio
    item survives as long as it was actually measurable."""
    store = MemoryStore()
    store.add_or_update_item("mildly_negative", "content", {"m": torch.tensor([1.0, 0.0])})
    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", lambda *a, **k: _record(-0.05))

    results = run_deep_scoring_pass(store, "fake-model", "cpu")  # keep_top_n defaults to None

    assert results["mildly_negative"]["outcome"] == "active"
    assert store.get_item("mildly_negative").status == PipelineStatus.ACTIVE


def test_run_deep_scoring_pass_prunes_unmeasurable_regardless_of_top_n(monkeypatch):
    """A None ratio means memorization_progress never even went
    positive - training didn't help the model predict its own content,
    a worse sign than a merely-negative-but-measurable ratio. Always
    pruned, even with a generous keep_top_n and even when it's the only
    candidate being scored."""
    store = MemoryStore()
    store.add_or_update_item("unmeasurable_topic", "content", {"m": torch.tensor([1.0, 0.0])})

    unmeasurable_record = DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=-1.0, generalization_mean=0.0,
        generalization_ratio_mean=None, generalization_ratio_stdev=None,
    )
    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", lambda *a, **k: unmeasurable_record)

    results = run_deep_scoring_pass(store, "fake-model", "cpu", keep_top_n=10)
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
