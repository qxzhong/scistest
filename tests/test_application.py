from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataApplication.run_framingham import (
    calibrate_observed,
    encode_sex,
    ordered_subsamples,
    seed_for,
)


def test_survset_sex_encodings_agree():
    for values in [["F", "M", "F"], [2, 1, 2], [0, 1, 0]]:
        np.testing.assert_array_equal(encode_sex(pd.Series(values)), [0, 1, 0])
    with pytest.raises(ValueError, match="categories"):
        encode_sex(pd.Series(["unknown"]))


def test_framingham_seed_and_subsampling_match_cluster():
    assert seed_for(0, 1, 0, 0) == 2330802983
    subsets = ordered_subsamples(4658, 1552, 10, seed_for(0, 9, 0, 0))
    assert subsets.shape == (30, 1552)
    for start in range(0, 30, 3):
        assert len(np.unique(subsets[start : start + 3])) == 4656


def test_application_log_tail_preserves_very_small_probabilities():
    result = calibrate_observed(np.array([1000.0] * 3), np.linspace(-1, 1, 30))
    assert np.isfinite(result["log_p_kde"])
    assert result["log_p_kde"] < -1000
    assert result["p_kde"] != "0"
    assert result["p_empirical"] == 0
