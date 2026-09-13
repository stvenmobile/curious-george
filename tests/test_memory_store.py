import torch

from curious_george.memory_store import (
    MemoryStore, PipelineStatus, DeepScoreRecord, classify_mastery,
    MASTERY_LOSS_CEILING, NOVICE_LOSS_CEILING,
)


def test_fresh_store_and_missing_load_path_are_empty(tmp_path):
    store = MemoryStore()
    assert len(store) == 0
    assert store.get_item("warbles") is None
    assert not store.has_topic("warbles")

    loaded_missing = MemoryStore.load(tmp_path / "never_saved")
    assert len(loaded_missing) == 0


def test_add_or_update_item_new_topic():
    store = MemoryStore()
    warble_vec = torch.tensor([1.0, 0.0, 0.0])
    item = store.add_or_update_item("warbles", "Warbles are green and yellow mammals.", {"method_a": warble_vec})

    assert len(store) == 1
    assert store.has_topic("warbles")
    assert item.content == "Warbles are green and yellow mammals."
    assert item.first_learned == item.last_studied
    assert item.loss_history == []
    assert torch.equal(store.embedding_matrices["method_a"][0], warble_vec)


def test_add_or_update_item_existing_topic_is_a_restudy_not_a_duplicate():
    store = MemoryStore()
    item = store.add_or_update_item("warbles", "Warbles are green and yellow mammals.", {"method_a": torch.tensor([1.0, 0.0, 0.0])})
    store.record_loss("warbles", 4.6)
    original_first_learned = item.first_learned

    new_vec = torch.tensor([0.0, 1.0, 0.0])
    updated_item = store.add_or_update_item("warbles", "Warbles are green and yellow mammals (updated).", {"method_a": new_vec})

    assert len(store) == 1, "re-studying an existing topic must not create a duplicate entry"
    assert updated_item.content == "Warbles are green and yellow mammals (updated)."
    assert updated_item.first_learned == original_first_learned, "first_learned must not change on a re-study"
    assert len(updated_item.loss_history) == 1, "re-studying must not touch existing loss history"
    assert torch.equal(store.embedding_matrices["method_a"][0], new_vec), "embedding row must update in place"


def test_record_loss_appends_and_raises_for_unknown_topic():
    store = MemoryStore()
    store.add_or_update_item("warbles", "content", {"method_a": torch.tensor([1.0, 0.0, 0.0])})
    store.record_loss("warbles", 4.6)
    store.record_loss("warbles", 3.1)

    assert store.get_item("warbles").loss_history[-1][1] == 3.1
    assert len(store.get_item("warbles").loss_history) == 2

    try:
        store.record_loss("quaddles", 5.0)
        raised = False
    except KeyError:
        raised = True
    assert raised, "recording loss for a never-studied topic should raise KeyError"


def test_learning_progress_none_with_one_measurement_and_correct_sign_with_two():
    store = MemoryStore()
    store.add_or_update_item("single_measurement_topic", "content", {"method_a": torch.tensor([1.0, 1.0, 1.0])})
    store.record_loss("single_measurement_topic", 2.0)
    assert store.learning_progress("single_measurement_topic") is None, "one measurement shouldn't yield a trend"

    store.add_or_update_item("improving_topic", "content", {"method_a": torch.tensor([1.0, 1.0, 1.0])})
    store.record_loss("improving_topic", 4.6)
    store.record_loss("improving_topic", 3.1)
    progress = store.learning_progress("improving_topic")
    assert abs(progress - 1.5) < 1e-9, f"expected improvement of 1.5, got {progress}"

    store.add_or_update_item("declining_topic", "content", {"method_a": torch.tensor([0.5, 0.5, 0.5])})
    store.record_loss("declining_topic", 1.0)
    store.record_loss("declining_topic", 2.5)  # got WORSE
    decline = store.learning_progress("declining_topic")
    assert decline < 0, f"expected a negative delta for worsening loss, got {decline}"


def test_query_ranks_by_cosine_similarity_and_handles_edge_cases():
    query_store = MemoryStore()
    query_store.add_or_update_item("close", "c", {"m": torch.tensor([1.0, 0.0])})
    query_store.add_or_update_item("orthogonal", "c", {"m": torch.tensor([0.0, 1.0])})
    query_store.add_or_update_item("opposite", "c", {"m": torch.tensor([-1.0, 0.0])})

    results = query_store.query(torch.tensor([1.0, 0.0]), "m", top_k=3)
    assert [topic for topic, _ in results] == ["close", "orthogonal", "opposite"], f"unexpected ranking: {results}"
    assert abs(results[0][1] - 1.0) < 1e-6
    assert abs(results[1][1] - 0.0) < 1e-6
    assert abs(results[2][1] - (-1.0)) < 1e-6

    assert MemoryStore().query(torch.tensor([1.0, 0.0]), "m", top_k=3) == []
    assert query_store.query(torch.tensor([1.0, 0.0]), "nonexistent_method", top_k=3) == []


def test_save_load_round_trip_preserves_everything_and_order(tmp_path):
    store = MemoryStore()
    store.add_or_update_item("warbles", "Warbles are green and yellow mammals.", {"method_a": torch.tensor([1.0, 0.0, 0.0])})
    store.record_loss("warbles", 4.6)
    store.record_loss("warbles", 3.1)
    store.add_or_update_item("quaddles", "Quaddles are blue and gray reptiles.", {"method_a": torch.tensor([0.0, 1.0, 0.0])})

    save_path = tmp_path / "roundtrip"
    store.save(save_path)
    reloaded = MemoryStore.load(save_path)

    assert len(reloaded) == len(store)
    assert [i.topic for i in reloaded.items] == [i.topic for i in store.items], "item order must survive the round-trip"

    for original, restored in zip(store.items, reloaded.items):
        assert original.topic == restored.topic
        assert original.content == restored.content
        assert original.first_learned == restored.first_learned
        assert original.last_studied == restored.last_studied
        assert original.loss_history == restored.loss_history

    for method, matrix in store.embedding_matrices.items():
        assert torch.equal(matrix, reloaded.embedding_matrices[method]), f"embedding matrix for {method!r} changed across save/load"


def test_save_load_round_trip_with_multiple_embedding_methods(tmp_path):
    multi_store = MemoryStore()
    multi_store.add_or_update_item("a", "content a", {
        "method_x": torch.tensor([1.0, 2.0]), "method_y": torch.tensor([9.0, 9.0, 9.0]),
    })
    multi_store.add_or_update_item("b", "content b", {
        "method_x": torch.tensor([3.0, 4.0]), "method_y": torch.tensor([8.0, 8.0, 8.0]),
    })
    multi_path = tmp_path / "multi_method"
    multi_store.save(multi_path)
    multi_reloaded = MemoryStore.load(multi_path)

    assert set(multi_reloaded.embedding_matrices.keys()) == {"method_x", "method_y"}
    assert torch.equal(multi_reloaded.embedding_matrices["method_x"], multi_store.embedding_matrices["method_x"])
    assert torch.equal(multi_reloaded.embedding_matrices["method_y"], multi_store.embedding_matrices["method_y"])


def test_restudy_with_unrecognized_embedding_method_raises():
    store = MemoryStore()
    store.add_or_update_item("a", "content a", {"method_x": torch.tensor([1.0, 2.0])})

    try:
        store.add_or_update_item("a", "content a again", {"method_z": torch.tensor([1.0])})
        raised = False
    except ValueError:
        raised = True
    assert raised, "re-study with a never-before-seen embedding method should raise, not silently misalign matrices"


def test_classify_mastery_uses_the_named_thresholds():
    assert classify_mastery(MASTERY_LOSS_CEILING - 0.1) == "master"
    assert classify_mastery(MASTERY_LOSS_CEILING) == "novice", "the ceiling itself belongs to the tier above it"
    assert classify_mastery(NOVICE_LOSS_CEILING - 0.1) == "novice"
    assert classify_mastery(NOVICE_LOSS_CEILING) == "newbie", "the ceiling itself belongs to the tier above it"
    assert classify_mastery(20.0) == "newbie"


def test_new_item_defaults_to_candidate_status_and_no_deep_score_history():
    store = MemoryStore()
    item = store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    assert item.status == PipelineStatus.CANDIDATE
    assert item.deep_score_history == []


def test_mastery_level_none_with_no_loss_history_then_tracks_latest_reading():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    assert store.mastery_level("topic") is None, "no loss measurement yet - nothing to classify"

    store.record_loss("topic", 8.0)
    assert store.mastery_level("topic") == "newbie"

    store.record_loss("topic", 0.5)
    assert store.mastery_level("topic") == "master", "mastery_level should track the MOST RECENT reading, not the first"


def test_set_status_updates_in_place():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    store.set_status("topic", PipelineStatus.ACTIVE)
    assert store.get_item("topic").status == PipelineStatus.ACTIVE

    try:
        store.set_status("nonexistent", PipelineStatus.ARCHIVED)
        raised = False
    except KeyError:
        raised = True
    assert raised, "setting status on an unknown topic should raise, not silently no-op"


def _deep_score(ratio_mean=0.25):
    return DeepScoreRecord(
        timestamp="2026-01-01T00:00:00+00:00", n_trials=5,
        memorization_mean=2.5, generalization_mean=0.5,
        generalization_ratio_mean=ratio_mean, generalization_ratio_stdev=0.05,
    )


def test_record_and_retrieve_deep_score_history():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    assert store.latest_deep_score("topic") is None, "no deep score recorded yet"

    first = _deep_score(ratio_mean=0.20)
    second = _deep_score(ratio_mean=0.30)
    store.record_deep_score("topic", first)
    store.record_deep_score("topic", second)

    assert store.get_item("topic").deep_score_history == [first, second], "append-only, like loss_history"
    assert store.latest_deep_score("topic") == second

    try:
        store.record_deep_score("nonexistent", first)
        raised = False
    except KeyError:
        raised = True
    assert raised, "recording a deep score for a never-added topic should raise"


def test_resonance_returns_max_cosine_similarity_to_declared_interests():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})

    interests = {
        "close_interest": torch.tensor([1.0, 0.0]),
        "orthogonal_interest": torch.tensor([0.0, 1.0]),
    }
    score = store.resonance("topic", "m", interests)
    assert abs(score - 1.0) < 1e-6, "should pick the CLOSEST interest, not average across all of them"


def test_resonance_returns_none_for_missing_method_or_no_interests():
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})

    assert store.resonance("topic", "nonexistent_method", {"x": torch.tensor([1.0, 0.0])}) is None
    assert store.resonance("topic", "m", {}) is None

    try:
        store.resonance("nonexistent_topic", "m", {"x": torch.tensor([1.0, 0.0])})
        raised = False
    except KeyError:
        raised = True
    assert raised, "resonance for an unknown topic should raise"


def test_remove_item_realigns_embeddings_and_index_for_remaining_items():
    store = MemoryStore()
    store.add_or_update_item("a", "content a", {"m": torch.tensor([1.0, 0.0])})
    store.add_or_update_item("b", "content b", {"m": torch.tensor([2.0, 0.0])})
    store.add_or_update_item("c", "content c", {"m": torch.tensor([3.0, 0.0])})

    store.remove_item("b")

    assert len(store) == 2
    assert not store.has_topic("b")
    assert [item.topic for item in store.items] == ["a", "c"], "remaining items must shift down, not leave a gap"
    assert torch.equal(store.embedding_matrices["m"], torch.tensor([[1.0, 0.0], [3.0, 0.0]])), (
        "embedding rows must stay positionally aligned with items after a removal"
    )
    # "c" must be reachable at its NEW index, not the stale old one
    assert store.get_item("c").content == "content c"

    try:
        store.remove_item("nonexistent")
        raised = False
    except KeyError:
        raised = True
    assert raised, "removing an unknown topic should raise, not silently no-op"


def test_save_load_round_trip_preserves_status_and_deep_score_history(tmp_path):
    store = MemoryStore()
    store.add_or_update_item("topic", "content", {"m": torch.tensor([1.0, 0.0])})
    store.set_status("topic", PipelineStatus.ARCHIVED)
    record = _deep_score(ratio_mean=0.42)
    store.record_deep_score("topic", record)

    save_path = tmp_path / "status_roundtrip"
    store.save(save_path)
    reloaded = MemoryStore.load(save_path)

    assert reloaded.get_item("topic").status == PipelineStatus.ARCHIVED
    assert reloaded.get_item("topic").deep_score_history == [record]


def test_load_tolerates_metadata_saved_before_this_schema_existed(tmp_path):
    """A store saved before status/deep_score_history existed has
    neither key in its metadata.json - must still load cleanly, with
    every item defaulting to CANDIDATE and an empty deep-score history,
    rather than raising a KeyError on old data."""
    import json
    import torch as torch_module

    save_path = tmp_path / "old_format"
    save_path.mkdir(parents=True)
    old_metadata = [{
        "topic": "old_topic", "content": "old content",
        "first_learned": "2025-01-01T00:00:00+00:00", "last_studied": "2025-01-01T00:00:00+00:00",
        "loss_history": [["2025-01-01T00:00:00+00:00", 3.0]],
    }]
    (save_path / "metadata.json").write_text(json.dumps(old_metadata), encoding="utf-8")
    torch_module.save({"m": torch.tensor([[1.0, 0.0]])}, save_path / "embeddings.pt")

    reloaded = MemoryStore.load(save_path)
    item = reloaded.get_item("old_topic")
    assert item.status == PipelineStatus.CANDIDATE
    assert item.deep_score_history == []
