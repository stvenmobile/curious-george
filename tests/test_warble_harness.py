from curious_george.validation_tools.warble_harness import with_context_probes


def test_with_context_probes_prepends_context_and_does_not_mutate_input():
    probes = [
        {"prompt": "Warbles are", "target": " green and yellow mammals."},
        {"prompt": "A group of warbles is called a", "target": " chorus."},
    ]

    result = with_context_probes(probes, "Some study context.")

    assert result[0]["prompt"] == "Some study context. Warbles are"
    assert result[0]["target"] == " green and yellow mammals."
    assert result[1]["prompt"] == "Some study context. A group of warbles is called a"
    assert result[1]["target"] == " chorus."

    assert probes[0]["prompt"] == "Warbles are", "with_context_probes must not mutate its input"
