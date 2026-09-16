import pytest

from curious_george.interest_pool import (
    InterestPool, current_value, interest_value, STARTING_VALUE, DECAY_PER_DAY, OPEN_SLOT_CAPACITY,
)
from curious_george.my_interests import PROTECTED_VALUE


def test_below_capacity_every_add_succeeds():
    pool = InterestPool()
    for i in range(OPEN_SLOT_CAPACITY):
        added = pool.add_interest(f"topic_{i}", f"description {i}", now="2026-01-01T00:00:00+00:00")
        assert added is True
    assert len(pool) == OPEN_SLOT_CAPACITY


def test_fresh_entry_starts_at_starting_value_with_no_decay():
    pool = InterestPool()
    pool.add_interest("topic", "description", now="2026-01-01T00:00:00+00:00")
    entry = pool.get("topic")
    assert current_value(entry, now="2026-01-01T00:00:00+00:00") == STARTING_VALUE


def test_value_decays_linearly_and_floors_at_zero():
    pool = InterestPool()
    pool.add_interest("topic", "description", now="2026-01-01T00:00:00+00:00")
    entry = pool.get("topic")

    assert current_value(entry, now="2026-01-11T00:00:00+00:00") == STARTING_VALUE - 10 * DECAY_PER_DAY
    assert current_value(entry, now="2026-03-15T00:00:00+00:00") == 0.0, "must floor at zero, not go negative"


def test_record_study_resets_decay_clock_to_full_value():
    pool = InterestPool()
    pool.add_interest("topic", "description", now="2026-01-01T00:00:00+00:00")

    # decay for 10 days, then study again
    pool.record_study("topic", now="2026-01-11T00:00:00+00:00")
    entry = pool.get("topic")
    assert current_value(entry, now="2026-01-11T00:00:00+00:00") == STARTING_VALUE, (
        "studying should reset value back to full, not just slow the decay"
    )

    try:
        pool.record_study("nonexistent", now="2026-01-11T00:00:00+00:00")
        raised = False
    except KeyError:
        raised = True
    assert raised, "recording study for an item never added should raise"


def test_add_interest_duplicate_name_raises():
    pool = InterestPool()
    pool.add_interest("topic", "description", now="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError):
        pool.add_interest("topic", "different description", now="2026-01-02T00:00:00+00:00")


def test_at_capacity_fresh_candidate_evicts_the_weakest_stale_entry():
    pool = InterestPool()
    for i in range(OPEN_SLOT_CAPACITY):
        pool.add_interest(f"topic_{i}", f"description {i}", now="2026-01-01T00:00:00+00:00")

    # topic_2 gets studied recently (still fresh); everything else decays for 30 days
    later = "2026-01-31T00:00:00+00:00"
    pool.record_study("topic_2", now=later)

    added = pool.add_interest("new_topic", "a fresh new idea", now=later)
    assert added is True
    assert pool.get("new_topic") is not None
    assert len(pool) == OPEN_SLOT_CAPACITY, "must still respect capacity - one slot freed, one filled"

    # every non-studied original topic decayed equally (30 days stale) and ties at the weakest -
    # exactly one of them was evicted, topic_2 (freshly studied) must have survived
    assert pool.get("topic_2") is not None, "the recently-studied entry should never be the weakest"


def test_at_capacity_all_entries_equally_fresh_rejects_new_candidate():
    pool = InterestPool()
    same_moment = "2026-01-01T00:00:00+00:00"
    for i in range(OPEN_SLOT_CAPACITY):
        pool.add_interest(f"topic_{i}", f"description {i}", now=same_moment)

    added = pool.add_interest("new_topic", "a fresh new idea", now=same_moment)
    assert added is False, "a tie (both at full starting value) should not evict the incumbent"
    assert pool.get("new_topic") is None
    assert len(pool) == OPEN_SLOT_CAPACITY


def test_save_load_round_trip(tmp_path):
    pool = InterestPool()
    pool.add_interest("topic", "description", now="2026-01-01T00:00:00+00:00")
    pool.record_study("topic", now="2026-01-05T00:00:00+00:00")

    save_path = tmp_path / "pool_roundtrip"
    pool.save(save_path)
    reloaded = InterestPool.load(save_path)

    assert len(reloaded) == 1
    entry = reloaded.get("topic")
    assert entry.description == "description"
    assert entry.anchor_since == "2026-01-05T00:00:00+00:00"


def test_load_missing_path_returns_empty_pool(tmp_path):
    pool = InterestPool.load(tmp_path / "never_saved")
    assert len(pool) == 0


def test_interest_value_returns_protected_value_for_one_of_the_original_five(monkeypatch):
    import curious_george.interest_pool as interest_pool_module

    def fake_is_protected(name, interests=None):
        return name == "human curiosity"

    monkeypatch.setattr("curious_george.my_interests.is_protected_interest", fake_is_protected)

    pool = InterestPool()
    assert interest_value("human curiosity", pool) == PROTECTED_VALUE


def test_interest_value_falls_back_to_pool_for_non_protected_names(monkeypatch):
    def fake_is_protected(name, interests=None):
        return False

    monkeypatch.setattr("curious_george.my_interests.is_protected_interest", fake_is_protected)

    pool = InterestPool()
    pool.add_interest("topic", "description", now="2026-01-01T00:00:00+00:00")
    assert interest_value("topic", pool, now="2026-01-01T00:00:00+00:00") == STARTING_VALUE


def test_interest_value_raises_for_a_name_in_neither_source(monkeypatch):
    def fake_is_protected(name, interests=None):
        return False

    monkeypatch.setattr("curious_george.my_interests.is_protected_interest", fake_is_protected)

    pool = InterestPool()
    with pytest.raises(KeyError):
        interest_value("nonexistent", pool)
