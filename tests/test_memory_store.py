import torch

from curious_george.memory_store import MemoryStore


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
