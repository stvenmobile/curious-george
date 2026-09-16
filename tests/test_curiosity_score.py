import torch

from curious_george.curiosity_tools.curiosity_score import curiosity_score, score_topic, rank_candidates
from curious_george.curiosity_tools.memory_store import MemoryStore, PipelineStatus, DeepScoreRecord


def _deep_score(generalization_mean, generalization_ratio_mean=None):
    return DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=2.0, generalization_mean=generalization_mean,
        generalization_ratio_mean=generalization_ratio_mean, generalization_ratio_stdev=0.05,
    )


def test_curiosity_score_none_without_a_deep_score():
    assert curiosity_score("novice", 0.5, None) is None


def test_curiosity_score_none_when_generalization_mean_is_none():
    record = DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=-1.0, generalization_mean=None,
        generalization_ratio_mean=None, generalization_ratio_stdev=None,
    )
    assert curiosity_score("novice", 0.5, record) is None


def test_curiosity_score_uses_raw_generalization_mean_as_the_base():
    record = _deep_score(generalization_mean=0.6389, generalization_ratio_mean=0.246)
    score = curiosity_score(mastery="novice", resonance=None, deep_score=record)
    assert abs(score - 0.6389) < 1e-9, "with no resonance and no mastery penalty, score should equal generalization_mean exactly"


def test_curiosity_score_adds_resonance_as_a_boost():
    record = _deep_score(generalization_mean=0.5)
    score_no_resonance = curiosity_score("novice", None, record)
    score_with_resonance = curiosity_score("novice", 0.76, record)
    assert score_with_resonance > score_no_resonance
    assert abs(score_with_resonance - (0.5 + 0.76)) < 1e-9


def test_curiosity_score_applies_mastery_penalty_only_for_master():
    record = _deep_score(generalization_mean=0.5)
    novice_score = curiosity_score("novice", None, record)
    newbie_score = curiosity_score("newbie", None, record)
    master_score = curiosity_score("master", None, record)

    assert novice_score == newbie_score == 0.5, "penalty should only apply to master, not novice/newbie"
    assert master_score < novice_score, "a fully-mastered topic should score lower - little room left to gain"
    assert master_score < 0, "the penalty should be large enough to sink a mastered topic below zero here"


def test_score_topic_pulls_mastery_resonance_and_deep_score_from_the_store():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    store.record_loss("topic", 3.0)  # -> novice
    store.record_deep_score("topic", _deep_score(generalization_mean=0.4))

    interests = {"some_interest": torch.tensor([1.0, 0.0])}  # identical vector -> resonance 1.0
    score = score_topic(store, "topic", interest_embeddings=interests, embedding_method="m")
    assert abs(score - (0.4 + 1.0)) < 1e-6


def test_score_topic_skips_resonance_when_interests_not_given():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    store.record_loss("topic", 3.0)
    store.record_deep_score("topic", _deep_score(generalization_mean=0.4))

    score = score_topic(store, "topic")
    assert abs(score - 0.4) < 1e-9


def test_score_topic_none_when_never_deep_scored():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    store.record_loss("topic", 3.0)
    assert score_topic(store, "topic") is None


def test_rank_candidates_sorts_highest_first_and_defaults_to_active_items():
    store = MemoryStore()

    store.add_or_update_item("best", "content", {"m": torch.tensor([1.0, 0.0])})
    store.record_loss("best", 3.0)
    store.record_deep_score("best", _deep_score(generalization_mean=0.6))
    store.set_status("best", PipelineStatus.ACTIVE)

    store.add_or_update_item("worst", "content", {"m": torch.tensor([0.0, 1.0])})
    store.record_loss("worst", 3.0)
    store.record_deep_score("worst", _deep_score(generalization_mean=-0.5))
    store.set_status("worst", PipelineStatus.ACTIVE)

    store.add_or_update_item("still_candidate", "content", {"m": torch.tensor([1.0, 1.0])})
    store.record_loss("still_candidate", 3.0)
    store.record_deep_score("still_candidate", _deep_score(generalization_mean=99.0))
    # status stays CANDIDATE - never promoted, should be excluded by default

    ranked = rank_candidates(store)
    assert [topic for topic, _ in ranked] == ["best", "worst"], (
        "should default to ACTIVE-status items only, ranked highest generalization_mean first"
    )


def test_rank_candidates_excludes_items_with_no_deep_score():
    store = MemoryStore()
    store.add_or_update_item("scored", "content", {"m": torch.tensor([1.0, 0.0])})
    store.record_loss("scored", 3.0)
    store.record_deep_score("scored", _deep_score(generalization_mean=0.1))
    store.set_status("scored", PipelineStatus.ACTIVE)

    store.add_or_update_item("unscored", "content", {"m": torch.tensor([0.0, 1.0])})
    store.record_loss("unscored", 3.0)
    store.set_status("unscored", PipelineStatus.ACTIVE)  # promoted somehow, but never actually deep-scored

    ranked = rank_candidates(store)
    assert [topic for topic, _ in ranked] == ["scored"]


def test_rank_candidates_respects_explicit_topics_list():
    store = MemoryStore()
    store.add_or_update_item("a", "content", {"m": torch.tensor([1.0, 0.0])})
    store.record_loss("a", 3.0)
    store.record_deep_score("a", _deep_score(generalization_mean=0.1))
    # never set ACTIVE - only reachable via explicit topics list

    ranked = rank_candidates(store, topics=["a"])
    assert [topic for topic, _ in ranked] == ["a"]
