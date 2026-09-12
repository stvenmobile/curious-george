from curious_george.curiosity import summarize_trials


def _trial(memorization, generalization, ratio):
    return {
        "memorization_progress": memorization,
        "generalization_progress": generalization,
        "generalization_ratio": ratio,
    }


def test_summarize_trials_computes_mean_and_stdev_for_multiple_trials():
    trials = [
        _trial(2.0, 1.0, 0.5),
        _trial(4.0, 3.0, 0.75),
        _trial(3.0, 2.0, 2 / 3),
    ]
    summary = summarize_trials(trials)

    assert summary["n"] == 3
    assert abs(summary["memorization_mean"] - 3.0) < 1e-9
    assert abs(summary["generalization_mean"] - 2.0) < 1e-9
    assert summary["memorization_stdev"] > 0
    assert summary["generalization_stdev"] > 0
    assert summary["ratio_n"] == 3


def test_summarize_trials_single_trial_has_zero_stdev_not_an_error():
    trials = [_trial(2.0, 1.0, 0.5)]
    summary = summarize_trials(trials)

    assert summary["n"] == 1
    assert summary["memorization_mean"] == 2.0
    assert summary["memorization_stdev"] == 0.0
    assert summary["generalization_stdev"] == 0.0
    assert summary["ratio_stdev"] == 0.0


def test_summarize_trials_excludes_none_ratios_from_the_average():
    trials = [
        _trial(2.0, 1.0, 0.5),
        _trial(-1.0, 0.5, None),  # memorization_progress <= 0 -> ratio is None, not 0
        _trial(4.0, 3.0, 0.75),
    ]
    summary = summarize_trials(trials)

    assert summary["ratio_n"] == 2, "the None-ratio trial must not count toward the ratio average"
    assert abs(summary["ratio_mean"] - 0.625) < 1e-9
    # memorization/generalization means DO include all 3 trials - only ratio is special-cased
    assert summary["n"] == 3
    assert abs(summary["memorization_mean"] - (2.0 - 1.0 + 4.0) / 3) < 1e-9


def test_summarize_trials_all_none_ratios_yields_none_mean():
    trials = [_trial(-1.0, 0.5, None), _trial(-2.0, 0.3, None)]
    summary = summarize_trials(trials)

    assert summary["ratio_n"] == 0
    assert summary["ratio_mean"] is None
    assert summary["ratio_stdev"] is None
