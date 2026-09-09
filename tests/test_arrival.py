import pytest

from replan.arrival import arrival_offsets, percentile


def test_batch_arrivals_are_simultaneous():
    assert arrival_offsets(3, mode="batch", rate=None, seed=1) == [0.0] * 3


def test_poisson_arrivals_are_reproducible_and_ordered():
    first = arrival_offsets(8, mode="poisson", rate=2.0, seed=7)
    second = arrival_offsets(8, mode="poisson", rate=2.0, seed=7)
    assert first == second
    assert first[0] == 0.0
    assert first == sorted(first)


def test_bursty_arrivals_group_parents():
    offsets = arrival_offsets(
        10, mode="bursty", rate=2.0, seed=7, burst_size=4
    )
    assert offsets[:4] == [0.0] * 4
    assert len(set(offsets[4:8])) == 1
    assert offsets[4] > 0.0


def test_non_batch_requires_rate():
    with pytest.raises(ValueError, match="positive arrival rate"):
        arrival_offsets(2, mode="poisson", rate=None, seed=1)


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
