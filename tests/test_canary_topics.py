from curious_george.curiosity_tools.canary_topics import load_canary_topics, get_canary_probes, all_canary_probes


def test_real_content_loads_with_expected_shape():
    topics = load_canary_topics()
    assert set(topics.keys()) == {"lighthouse_keeping", "vintage_typewriters", "traditional_knots"}

    for name in topics:
        probes = get_canary_probes(topics, name)
        assert len(probes) == 8, f"{name}: expected 8 probes"
        for probe in probes:
            assert set(probe.keys()) == {"prompt", "target"}, f"malformed probe in {name}: {probe}"
            assert probe["target"].startswith(" "), f"probe target should start with a space: {probe}"


def test_all_canary_probes_pools_every_topic_together():
    topics = load_canary_topics()
    combined = all_canary_probes(topics)
    assert len(combined) == 24, "3 topics x 8 probes each"

    lighthouse_probes = get_canary_probes(topics, "lighthouse_keeping")
    assert lighthouse_probes[0] in combined


def test_canary_domains_share_no_thematic_vocabulary_with_declared_interests():
    """Loose sanity check, not a strict proof: the canary domains were
    deliberately picked to be offbeat and unrelated to the declared
    interests (human curiosity, agentic behavior, goal prioritization,
    memory as a predictor, free will) - this just confirms no literal
    word overlap between the two content sets, catching an accidental
    thematic collision if one were ever introduced."""
    from curious_george.curiosity_tools.my_interests import load_my_interests

    canary_words = set()
    for probe in all_canary_probes(load_canary_topics()):
        canary_words.update(w.strip(".,").lower() for w in (probe["prompt"] + probe["target"]).split())

    interest_words = set()
    for entry in load_my_interests():
        interest_words.update(w.strip(".,").lower() for w in entry["description"].lower().split())

    # allow common function words to overlap; only flag substantial (4+ letter) content-word collisions
    substantial_overlap = {w for w in (canary_words & interest_words) if len(w) >= 5}
    assert not substantial_overlap, f"unexpected thematic overlap between canaries and declared interests: {substantial_overlap}"
