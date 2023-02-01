"""
Library to clean the data including:
Noise elimation 
Filter frequency to 18-90kHz 

"""
from copy import deepcopy
import numpy as np
from scipy.signal import medfilt2d, lfilter, butter, sosfilt


def bandpassButter(
    signal: np.ndarray,
    order: int = 2,
    sr: int = 384000,
    min: int = 18000,
    max: int = 100000,
) -> np.ndarray:
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

    return cleaned


def bandpassSOS(
    signal: np.ndarray,
    order: int = 2,
    sr: int = 384000,
    minCritfreq: int = 18000,
    maxCritfreq: int = 100000,
) -> np.ndarray:
    """Applies bandpass filter to raw_buffer between 18000,100000

    Args:
        signal (np.ndarray): raw_buffer of audio
        order (int, optional): roll off for the bandpass filter. Defaults to 2.
        sr (int, optional): sample rate of the aduio. Defaults to 384000.
        min (int, optional): minimum frequency bound of bandpass. Defaults to 18000.
        max (int, optional): maximum frequency bound of bandpass. Defaults to 100000.

    Returns:
        np.ndarray: bandpassed audio
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
    print(sos.shape)
    _signal = deepcopy(signal)
    print(_signal.shape)
    cleaned = sosfilt(sos=sos, x=_signal, zi=None)
    print(cleaned.shape)
    return cleaned


def local_median_filter(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    we applied a Local Median Filter step,
    a method to estimate the minimum expected contrast between
    a USV and its background for each audio recording


    This is done after finding candidate USV's from the audio data.

    Takes input np.array as input and kernel size, default is 5

    """

    local_filt = medfilt2d(signal, kernel_size)
    return local_filt


def drop_back2(spectrogram_list: list[np.ndarray], threshold: int = 20) -> np.ndarray:
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


def drop_back(spectrograms: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Cleaning function replaces spectrogram points below a certain threshold.
    Applied to a single spectrogram
    specs: spectrogram of audio data
    threshold: decible threshold default 20
    returns cleaned version of original spectrogram
    """

    threshold = -threshold
    cleaned = list()

    minima = -80

    for row in spectrograms:
        cleaned.append(np.where(row <= threshold, minima, row))
    cleaned = np.array(cleaned)
    return cleaned


def remove_empty_points(points_array: np.ndarray) -> np.ndarray:
    """
    Cleaning function that removes points that were converted to minima
    """

    # make list
    points_list = points_array.tolist()

    # find elements with rows of only -80
    for index, point in enumerate(points_list):
        # if row only has -80 values
        drop_test = all(point)
        if drop_test:
            del points_list[index]
        else:
            pass

    points_list = np.array(points_list)

    return points_list
