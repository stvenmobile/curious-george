import pytest

from curious_george.validation_tools.curiosity_topics import (
    load_curiosity_topics, get_category, get_train_facts, get_trained_probes,
    get_sibling_facts, get_sibling_probes, _validate_no_train_sibling_overlap,
)


def test_real_content_loads_with_expected_categories_and_shape():
    topics = load_curiosity_topics()
    assert set(topics.keys()) == {"known", "moderate", "noise"}

    for name in ("known", "moderate", "noise"):
        train_facts = get_train_facts(topics, name)
        trained_probes = get_trained_probes(topics, name)
        sibling_facts = get_sibling_facts(topics, name)
        sibling_probes = get_sibling_probes(topics, name)

        assert len(trained_probes) == 8, f"{name}: expected 8 trained probes"
        assert len(sibling_probes) == 8, f"{name}: expected 8 sibling probes"
        assert len(sibling_facts) > 0, f"{name}: sibling_facts should document what the sibling topic is"
        for probe in trained_probes + sibling_probes:
            assert set(probe.keys()) == {"prompt", "target"}, f"malformed probe in {name}: {probe}"
            assert probe["target"].startswith(" "), f"probe target should start with a space: {probe}"


def test_categories_are_labeled_as_expected():
    topics = load_curiosity_topics()
    assert get_category(topics, "known") == "known"
    assert get_category(topics, "moderate") == "moderate"
    assert get_category(topics, "noise") == "noise"


def test_train_and_sibling_facts_are_disjoint_for_every_topic():
    topics = load_curiosity_topics()
    for name in ("known", "moderate", "noise"):
        train_facts = set(get_train_facts(topics, name))
        sibling_facts = set(get_sibling_facts(topics, name))
        assert train_facts.isdisjoint(sibling_facts), (
            f"{name}: a fact in both train_facts and sibling_facts means the sibling was trained on"
        )


def test_validation_raises_when_a_fact_is_in_both_train_and_sibling():
    broken_topics = {
        "broken": {
            "category": "known",
            "train_facts": ["Shared fact.", "Only trained."],
            "trained_probes": [],
            "sibling_facts": ["Shared fact.", "Only sibling."],
            "sibling_probes": [],
        }
    }
    with pytest.raises(ValueError):
        _validate_no_train_sibling_overlap(broken_topics)


def test_noise_topic_shares_no_word_between_train_and_sibling_facts():
    """The specific leak an earlier version of this content had: the
    original noise category reused ~25 words across all its sentences,
    so training on train_facts raised the model's probability on words
    the "held out" sentences happened to share too - real loss
    improvement, but from vocabulary overlap, not from any structure the
    noise category was meant to lack. Word-level disjointness (not just
    whole-sentence disjointness, already checked above) is what actually
    rules that out for the sibling design too."""
    topics = load_curiosity_topics()
    train_words = set(
        word.strip(".,").lower()
        for fact in get_train_facts(topics, "noise")
        for word in fact.split()
    )
    sibling_words = set(
        word.strip(".,").lower()
        for fact in get_sibling_facts(topics, "noise")
        for word in fact.split()
    )
    overlap = train_words & sibling_words
    assert not overlap, f"noise topic's train and sibling facts share word(s): {overlap}"


def test_moderate_topic_matches_phase_0s_original_warbles_and_quaddles():
    """Cross-check against Phase 0's already-validated content in
    piper_assistant/feature/piper-memory - moderate's train_facts should
    be exactly the 10 original warble facts, and its sibling should be
    exactly the original quaddle content, not an independent copy that
    could have drifted."""
    from curious_george.validation_tools.fictional_entities import load_fictional_entities, get_study_facts, get_recall_probes

    topics = load_curiosity_topics()
    entities = load_fictional_entities()

    # quaddles is control-role and get_study_facts() correctly refuses to hand out
    # control-role facts as study material - reading entities["quaddles"] directly
    # here is fine, since this comparison never trains on them, only verifies the copy.
    assert set(get_train_facts(topics, "moderate")) == set(get_study_facts(entities, "warbles"))
    assert set(get_sibling_facts(topics, "moderate")) == set(entities["quaddles"]["study_facts"])
    assert get_sibling_probes(topics, "moderate") == get_recall_probes(entities, "quaddles")
