import json

from curious_george.validation_tools.fictional_entities import (
    load_fictional_entities, get_study_facts, get_recall_probes, get_generalization_probes,
)


def test_real_content_file_loads_with_expected_shape():
    entities = load_fictional_entities()
    assert set(entities.keys()) == {"warbles", "quaddles"}
    assert entities["warbles"]["role"] == "studied"
    assert entities["quaddles"]["role"] == "control"

    for name in ("warbles", "quaddles"):
        assert len(entities[name]["study_facts"]) == 10, f"{name} should have 10 study facts"
        assert len(entities[name]["recall_probes"]) == 8, f"{name} should have 8 recall probes"
        assert len(entities[name]["generalization_probes"]) == 4, f"{name} should have 4 generalization probes"
        for probe in entities[name]["recall_probes"]:
            assert set(probe.keys()) == {"prompt", "target"}, f"malformed recall probe in {name}: {probe}"
            assert probe["target"].startswith(" "), f"recall probe target should start with a space for clean tokenization: {probe}"


def test_get_study_facts_impossible_for_control_role_entity():
    entities = load_fictional_entities()
    try:
        get_study_facts(entities, "quaddles")
        raised = False
    except ValueError:
        raised = True
    assert raised, "pulling study facts for the control entity (quaddles) must raise, not silently succeed"

    warble_facts = get_study_facts(entities, "warbles")
    assert len(warble_facts) == 10
    assert all(isinstance(f, str) and f for f in warble_facts)


def test_probes_accessible_for_both_roles():
    entities = load_fictional_entities()
    for name in ("warbles", "quaddles"):
        recall = get_recall_probes(entities, name)
        generalization = get_generalization_probes(entities, name)
        assert len(recall) == 8
        assert len(generalization) == 4


def test_warbles_and_quaddles_share_no_identical_fact_text():
    entities = load_fictional_entities()
    warble_fact_set = set(entities["warbles"]["study_facts"])
    quaddle_fact_set = set(entities["quaddles"]["study_facts"])
    assert warble_fact_set.isdisjoint(quaddle_fact_set), "warbles and quaddles must not share identical fact text"


def test_loader_works_against_custom_path_and_safety_check_holds(tmp_path):
    fake = {
        "florbs": {"role": "studied", "study_facts": ["Florbs are purple."],
                   "recall_probes": [{"prompt": "Florbs are", "target": " purple."}],
                   "generalization_probes": []},
        "grelks": {"role": "control", "study_facts": ["Grelks are orange."],
                   "recall_probes": [{"prompt": "Grelks are", "target": " orange."}],
                   "generalization_probes": []},
    }
    tmp_path_file = tmp_path / "fake_fictional_entities.json"
    tmp_path_file.write_text(json.dumps(fake), encoding="utf-8")

    fake_entities = load_fictional_entities(path=tmp_path_file)
    assert get_study_facts(fake_entities, "florbs") == ["Florbs are purple."]
    try:
        get_study_facts(fake_entities, "grelks")
        raised = False
    except ValueError:
        raised = True
    assert raised, "the safety check must work against any loaded file, not just the bundled default"
