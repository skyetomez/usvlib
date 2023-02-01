import os
import sys
import pathlib

import numpy as np


def get_time_bins(time_array: np.ndarray, width: int = 20) -> np.ndarray:
    """
    Creates indexes of time bins for the separation of the spectrograms
    """
    threshold = width // 1000  # ms to seconds

    store = dict()
    time_bins = list()

    for index, point in enumerate(time_array):
        store[point] = index

        if len(store.keys()) >= 2:
            if max(store.keys()) - min(store.keys()) > threshold:
                time_bins.append(list(store.values()))
                store = dict()

    time_bins = np.array(list(map(np.array, time_bins)))
    return time_bins
