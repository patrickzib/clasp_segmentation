# -*- coding: utf-8 -*-
"""Simple ClaSP test."""

__author__ = ["patrickzib"]
__all__ = []

import numpy as np
import pandas as pd

from clasp.annotation.clasp import ClaSPSegmentation, find_dominant_window_sizes
from sktime.datasets import load_gun_point_segmentation


def test_clasp_sparse():
    """Test ClaSP sparse segmentation.

    Check if the predicted change points match.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, n_cps=1)
    clasp.fit(ts)
    found_cps = clasp.predict(ts)
    scores = clasp.predict_scores(ts)

    assert len(found_cps) == 1 and found_cps[0] == 893
    assert len(scores) == 1 and scores[0] > 0.74


def test_clasp_dense():
    """Tests ClaSP dense segmentation.

    Check if the predicted segmentation matches.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, n_cps=1, fmt="dense")
    clasp.fit(ts)
    segmentation = clasp.predict(ts)
    scores = clasp.predict_scores(ts)

    assert len(segmentation) == 2 and segmentation[0].right == 893
    assert np.argmax(scores) == 893


def test_clasp_activity():
    def load_data():
        np_cols = ["x-acc", "y-acc", "z-acc", "x-gyro", "y-gyro",
                   "z-gyro", "x-mag", "y-mag", "z-mag", "change_points",
                   "activities", "lat", "lon", "speed"]
        converters = {col: lambda val: None if len(val) == 0 else np.array(eval(val))
                      for col in np_cols}
        return pd.read_csv("/Users/bzcschae/workspace/activity_challenge/har_challenge_master.csv.gz",
                           converters=converters,
                           compression="gzip")

    df = load_data()

    dataset = df[["x-acc", "change_points"]].iloc[213:214]

    ts = dataset["x-acc"].values[0]
    true_cps = dataset["change_points"].values[0]
    m = find_dominant_window_sizes(ts) * 2
    clasp = ClaSPSegmentation(period_length=m, n_cps=5, fmt="sparse")
    found_cps = clasp.fit_predict(ts)

    print("i", # i + 1,
          "Period-Size", m,
          "Found", found_cps,
          "True", true_cps
          )
