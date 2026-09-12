import pytest

from curious_george.curiosity_topics import (
    load_curiosity_topics, get_category, get_train_facts, get_trained_probes,
    get_held_out_facts, get_held_out_probes, _validate_no_train_held_out_overlap,
)


def test_real_content_loads_with_expected_categories_and_shape():
    topics = load_curiosity_topics()
    assert set(topics.keys()) == {"known", "moderate", "noise"}

    for name in ("known", "moderate", "noise"):
        train_facts = get_train_facts(topics, name)
        trained_probes = get_trained_probes(topics, name)
        held_out_facts = get_held_out_facts(topics, name)
        held_out_probes = get_held_out_probes(topics, name)

        assert len(train_facts) == len(trained_probes) == 8, f"{name}: expected 8 trained facts/probes"
        assert len(held_out_facts) == len(held_out_probes) == 2, f"{name}: expected 2 held-out facts/probes"
        for probe in trained_probes + held_out_probes:
            assert set(probe.keys()) == {"prompt", "target"}, f"malformed probe in {name}: {probe}"
            assert probe["target"].startswith(" "), f"probe target should start with a space: {probe}"


def test_categories_are_labeled_as_expected():
    topics = load_curiosity_topics()
    assert get_category(topics, "known") == "known"
    assert get_category(topics, "moderate") == "moderate"
    assert get_category(topics, "noise") == "noise"


def test_train_and_held_out_facts_are_disjoint_for_every_topic():
    topics = load_curiosity_topics()
    for name in ("known", "moderate", "noise"):
        train_facts = set(get_train_facts(topics, name))
        held_out_facts = set(get_held_out_facts(topics, name))
        assert train_facts.isdisjoint(held_out_facts), (
            f"{name}: a fact in both train_facts and held_out_facts invalidates the generalization measurement"
        )


def test_validation_raises_when_a_fact_is_in_both_train_and_held_out():
    broken_topics = {
        "broken": {
            "category": "known",
            "train_facts": ["Shared fact.", "Only trained."],
            "trained_probes": [],
            "held_out_facts": ["Shared fact.", "Only held out."],
            "held_out_probes": [],
        }
    }
    with pytest.raises(ValueError):
        _validate_no_train_held_out_overlap(broken_topics)


def test_moderate_topic_reconstructs_the_original_ten_warble_facts():
    """Cross-check against Phase 0's already-validated warble content -
    catches a transcription error if the 8 train + 2 held-out facts
    don't add back up to exactly the original 10."""
    from curious_george.fictional_entities import load_fictional_entities, get_study_facts

    topics = load_curiosity_topics()
    moderate_facts = set(get_train_facts(topics, "moderate")) | set(get_held_out_facts(topics, "moderate"))

    entities = load_fictional_entities()
    original_warble_facts = set(get_study_facts(entities, "warbles"))

    assert moderate_facts == original_warble_facts, "moderate topic's facts must exactly match Phase 0's warbles"
