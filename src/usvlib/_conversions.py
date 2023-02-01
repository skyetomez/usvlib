import os
import sys
import pathlib

import numpy as np
from typing import Tuple


stft_parameters = {
    "window": "hann",  # window type
    "nperseg": 256,
    "noverlap": 128,
    "nfft": 256,
    "padded": True,
    "axis": -1,
    "sample_rate": 384000,  # cyceles/sec
    "resolution": 20,  # ms
    "n_fft": "int(sample_rate / time_res)",
}


def get_time_segments(spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    freq_dim, time_dim = spec.shape
    upper_freq = int(stft_parameters["nfft"] // 2 + 1)

    ms_segment = get_ms_segments()

    time_array = np.array([ms_segment * n for n in range(time_dim)])
    freq_array = np.linspace(0, upper_freq, freq_dim)

    return freq_array, time_array


def get_ms_segments() -> int:
    time_per_segment_ms = stft_parameters["resolution"]
    nperseg = int(stft_parameters["sample_rate"] * 0.001 * time_per_segment_ms)
    noverlap = nperseg // 4
    seconds_per_segment = (stft_parameters["nperseg"] - noverlap) / stft_parameters[
        "sample_rate"
    ]
    ms_per_segment = int(seconds_per_segment * 1000)
    return ms_per_segment


def points_array_to_dict(data: np.ndarray) -> dict:
    """
    Takes an array and returns a dictionary of points and
    values
    """
    # flatten data
    flat_data = list()

    for section in data:
        tmp = section.flatten()
        flat_data.append(tmp.tolist())

    flat_data_labels = list(range((len(flat_data))))

    # create label dict
    data_dict = dict()
    iter_object = zip(flat_data_labels, flat_data)
    for k, v in list(iter_object):
        data_dict[k] = v

    return data_dict
