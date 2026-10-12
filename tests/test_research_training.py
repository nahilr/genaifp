"""Tests for group isolation and calibration behavior."""

import pandas as pd
import pytest

from genaifp.data_pipeline import TARGET_LABELS
from genaifp.research import _select_group_splits


def _grouped_rows():
    rows = []
    for label in TARGET_LABELS:
        for group_index in range(20):
            group_id = f"{label}-{group_index}"
            rows.append({"label": label, "is_ood": False, "group_id": group_id})
            rows.append({"label": label, "is_ood": True, "group_id": group_id})
    for group_index in range(20):
        rows.append({"label": "unknown", "is_ood": True, "group_id": f"unknown-{group_index}"})
    return pd.DataFrame(rows)


def test_splitter_keeps_all_related_samples_in_one_partition():
    frame = _grouped_rows()
    train, calibration, test = _select_group_splits(frame, 42)
    group_sets = [set(frame.iloc[indices]["group_id"]) for indices in (train, calibration, test)]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    for indices in (train, calibration, test):
        known = frame.iloc[indices].loc[lambda rows: ~rows["is_ood"], "label"]
        assert set(TARGET_LABELS).issubset(set(known))
    assert frame.iloc[calibration]["is_ood"].any()
    assert frame.iloc[test]["is_ood"].any()


def test_splitter_fails_when_group_count_cannot_support_holdouts():
    frame = _grouped_rows()
    frame = frame[frame["group_id"].str.endswith("-0")]
    with pytest.raises(ValueError, match="independent groups|Could not create"):
        _select_group_splits(frame, 42)
