"""
Library to clean the data including:
Noise elimation 
Filter frequency to 18-90kHz 

"""
from copy import deepcopy
import numpy as np
from scipy.signal import medfilt2d, lfilter, butter, sosfilt


from ._type_annotations import *


def bandpassButter(
    signal: NDArray,
    order: int = 2,
    sr: int = 384000,
    min: int = 18000,
    max: int = 100000,
) -> NDArray:
    """
    Bandpass filter between 18kHz and 100kHz by default
    signal: numpy array
    min:  lower bound of bandpass filter
    max: upper bound of bandpass filter
    sr: sample_rate of input signal
    """
    assert max >= 0, "Upper bound for frequency must be provided"
    assert min >= 0, "Lower bound for frequency must be provided"
    b, a = butter(order, [min, max], btype="bandpass", analog=False, output="ba", fs=sr)

    _signal = deepcopy(signal)
    cleaned = lfilter(b, a, _signal, axis=-1, zi=None)

    return cleaned  # type: ignore


def bandpassSOS(
    signal: NDArray,
    order: int = 2,
    sr: int = 384000,
    minCritfreq: int = 18000,
    maxCritfreq: int = 100000,
) -> NDArray:
    """Applies bandpass filter to raw_buffer between 18000,100000

    Args:
        signal (NDArray): raw_buffer of audio
        order (int, optional): roll off for the bandpass filter. Defaults to 2.
        sr (int, optional): sample rate of the aduio. Defaults to 384000.
        min (int, optional): minimum frequency bound of bandpass. Defaults to 18000.
        max (int, optional): maximum frequency bound of bandpass. Defaults to 100000.

    Returns:
        NDArray: bandpassed audio
    """

    assert maxCritfreq >= 0, "Upper bound for frequency must be provided"
    assert minCritfreq >= 0, "Lower bound for frequency must be provided"

    sos = butter(
        N=order,
        Wn=(minCritfreq, maxCritfreq),
        btype="bandpass",
        analog=False,
        output="sos",
        fs=sr,
    )
    _signal = deepcopy(signal)
    cleaned = sosfilt(sos=sos, x=_signal, zi=None)

    return cleaned  # type: ignore


def local_median_filter(signal: NDArray, kernel_size: int = 5) -> NDArray:
    """
    we applied a Local Median Filter step,
    a method to estimate the minimum expected contrast between
    a USV and its background for each audio recording


    This is done after finding candidate USV's from the audio data.

    Takes input np.array as input and kernel size, default is 5

    """

    local_filt = medfilt2d(signal, kernel_size)
    return local_filt


def drop_back2(spectrogram_list: list[NDArray], threshold: int = 20) -> NDArray:
    """
    Cleaning function replaces spectrogram points below a certain threshold.
    specs: spectrogram of audio data
    threshold: decible threshold default 20

    returns cleaned version of original spectrograms as new array.
    """
    if threshold > 0:
        threshold = -threshold

    cleaned = list()
    minima = -80  # quiet

    n_fft, t_dim = spectrogram_list[0].shape

    for spectrogram in spectrogram_list:
        tmp = deepcopy(spectrogram)

        for coord in range(t_dim):
            tmp[:, coord] = np.where(tmp[:, coord] >= threshold, tmp[:, coord], minima)

        cleaned.append(tmp)

    cleaned = np.array(cleaned)

    return cleaned


def drop_back(spectrogram: NDArray) -> NDArray:
    """
    Cleaning function replaces spectrogram points below a certain threshold.
    Applied to a single spectrogram
    specs: spectrogram of audio data
    threshold: decible threshold default 20
    returns cleaned version of original spectrogram
    """

    col_avg = np.nanmean(spectrogram, axis=0)
    col_max = np.ceil(np.max(col_avg))  # ceil to be more sensitive

    minima = -80

    cleaned = np.where(spectrogram.T >= col_max, spectrogram.T, minima)

    return cleaned.T


def remove_empty_points(
    points_array: NDArray, time_arr: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    Cleaning function that removes points that were converted to minima

    Returns compressed spectrogram and time array
    """

    # make list
    if isinstance(points_array, NpzFile):
        arr = points_array["arr_0"]
    else:
        arr = points_array

    # find elements with rows of only -80
    indices = list()

    for idx, point in enumerate(arr.T):
        # if col only has -80 values
        if np.equal(point, -80).all():
            indices.append(idx)

    return np.delete(arr, indices, axis=1), np.delete(time_arr, indices, axis=-1)
