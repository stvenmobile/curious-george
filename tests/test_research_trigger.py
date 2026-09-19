import json

import torch

from curious_george.curiosity_tools import research_trigger
from curious_george.curiosity_tools.memory_store import DeepScoreRecord, MemoryStore, PipelineStatus


def _record(generalization_mean):
    return DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=1.0, generalization_mean=generalization_mean,
        generalization_ratio_mean=0.2, generalization_ratio_stdev=0.05,
    )


def _add_active(store, topic, loss, last_studied=None, deep_score=None):
    store.add_or_update_item(topic, f"content for {topic}", {"m": torch.tensor([1.0, 0.0])})
    store.record_loss(topic, loss)
    store.set_status(topic, PipelineStatus.ACTIVE)
    if deep_score is not None:
        store.record_deep_score(topic, deep_score)
    if last_studied is not None:
        store.mark_studied(topic, now=last_studied)
    return store


def test_sanitize_short_name_cleans_and_truncates():
    assert research_trigger._sanitize_short_name("Goal Arbitration!!") == "goal_arbitra"
    assert research_trigger._sanitize_short_name("already_short") == "already_shor"
    assert research_trigger._sanitize_short_name("   ") == "topic"


def test_select_next_topic_returns_none_when_no_active_topics():
    store = MemoryStore()
    store.add_or_update_item("only_candidate", "content", {"m": torch.tensor([1.0, 0.0])})
    assert research_trigger.select_next_topic(store) is None


def test_select_next_topic_picks_least_recently_studied_newbie():
    store = MemoryStore()
    _add_active(store, "newbie_recent", loss=8.0, last_studied="2026-09-18T00:00:00+00:00")
    _add_active(store, "newbie_stale", loss=7.0, last_studied="2026-09-01T00:00:00+00:00")
    _add_active(store, "already_master", loss=0.5, last_studied="2026-01-01T00:00:00+00:00")

    assert research_trigger.select_next_topic(store) == "newbie_stale", (
        "breadth phase should rotate to the newbie studied longest ago, not the higher-scoring master"
    )


def test_select_next_topic_picks_top_curiosity_score_when_no_newbies_remain():
    store = MemoryStore()
    _add_active(store, "novice_weak", loss=4.0, deep_score=_record(-0.5))
    _add_active(store, "novice_strong", loss=4.0, deep_score=_record(0.8))

    assert research_trigger.select_next_topic(store) == "novice_strong", (
        "with no newbies left, depth phase should defer to curiosity_score ranking"
    )


def test_enforce_active_capacity_shelves_weakest_beyond_capacity():
    store = MemoryStore()
    for name, score in [("a", 0.9), ("b", 0.7), ("c", 0.5), ("d", 0.3), ("e", 0.1)]:
        _add_active(store, name, loss=3.0, deep_score=_record(score))

    shelved = research_trigger._enforce_active_capacity(store, capacity=3)

    assert set(shelved) == {"d", "e"}
    assert store.get_item("d").status == PipelineStatus.SHELVED
    assert store.get_item("e").status == PipelineStatus.SHELVED
    assert store.get_item("a").status == PipelineStatus.ACTIVE
    assert store.get_item("c").status == PipelineStatus.ACTIVE
    assert store.get_item("d").deep_score_history, "shelving must not discard deep-score history"


class FakeEmbedder:
    METHOD_NAME = "fake_method"

    def embed(self, text):
        return torch.tensor([1.0, 0.0])


def test_promote_candidates_ingests_suggestions_deep_scores_and_enforces_capacity(tmp_path, monkeypatch):
    suggestions_path = tmp_path / "suggestions.json"
    suggestions_path.write_text(json.dumps([
        {"topic": "brand_new", "content": "some fresh candidate content"},
    ]), encoding="utf-8")

    store = MemoryStore()
    for name, score in [("existing_a", 0.9), ("existing_b", 0.7), ("existing_c", 0.5),
                         ("existing_d", 0.3), ("existing_e", 0.1)]:
        _add_active(store, name, loss=3.0, deep_score=_record(score))

    class FakeModel:
        def __call__(self, input_ids, attention_mask, labels):
            class Out:
                loss = torch.tensor(2.0)
            return Out()

    class FakeTokenizer:
        def __call__(self, text, return_tensors="pt", add_special_tokens=False):
            class Batch(dict):
                def to(self, device):
                    return self
                @property
                def input_ids(self):
                    return self["input_ids"]
            return Batch(input_ids=torch.tensor([[1, 2, 3]]))

    import curious_george.curiosity_tools.deep_scoring as deep_scoring_module
    monkeypatch.setattr(deep_scoring_module, "deep_score_topic", lambda *a, **k: _record(0.95))

    result = research_trigger.promote_candidates(
        store, FakeModel(), FakeTokenizer(), "fake-model", "cpu", FakeEmbedder(),
        suggestions_path=suggestions_path,
    )

    assert result["suggestions_considered"] == 1
    assert store.has_topic("brand_new"), "the new suggestion should have been added as a candidate"
    assert store.get_item("brand_new").status == PipelineStatus.ACTIVE, (
        "it scored well (0.95) and should have been promoted by deep-scoring"
    )
    assert result["shelved"] == ["existing_e"], (
        "6 active topics now compete for ACTIVE_CAPACITY=5 slots - the weakest (existing_e) should be shelved"
    )
    assert store.get_item("existing_e").status == PipelineStatus.SHELVED


def test_run_research_trigger_topic_override_skips_promotion_and_writes_log(tmp_path, monkeypatch):
    store = MemoryStore()
    _add_active(store, "manual_topic", loss=3.0, deep_score=_record(0.4))

    import curious_george.curiosity_tools.study as study_module
    monkeypatch.setattr(study_module, "study_topic", lambda store, model_name, device, topic, **kw: {
        "topic": topic, "loss_before": 3.0, "loss_after": 1.0,
        "progress_this_session": 2.0, "mastery": "master",
    })

    result = research_trigger.run_research_trigger(
        store, model=None, tokenizer=None, model_name="fake-model", device="cpu", embedder=FakeEmbedder(),
        topic="manual_topic", num_steps=50, log_dir=tmp_path,
    )

    assert result["outcome"] == "studied"
    assert result["topic"] == "manual_topic"
    assert result["promotion"] is None, "an override should skip promote_candidates entirely"

    log_path = tmp_path / f"{result['short_name']}_{result['timestamp'].replace(':', '-')}.json"
    assert log_path.exists()
    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged["topic"] == "manual_topic"
    assert logged["num_steps"] == 50


def test_run_research_trigger_returns_nothing_to_study_on_an_empty_store(tmp_path):
    suggestions_path = tmp_path / "suggestions.json"
    suggestions_path.write_text("[]", encoding="utf-8")
    store = MemoryStore()

    class FakeModel:
        def __call__(self, *a, **k):
            raise AssertionError("no candidates to score - the model should never be called")

    result = research_trigger.promote_candidates(
        store, FakeModel(), None, "fake-model", "cpu", FakeEmbedder(), suggestions_path=suggestions_path,
    )
    assert result["new_candidates"] == []
    assert research_trigger.select_next_topic(store) is None
