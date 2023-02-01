import os
import sys
import pathlib


import numpy as np
from ssqueezepy import ssq_cwt

from typing import Tuple
from scipy.io import wavfile
from scipy.signal import stft
from pyflac import FileDecoder


import pickle5


"""
Every function name in this library works on a path,
a path as a string or a filename. These functions also read from disk to memory
"""


stft_parameters = {
    "window": "hann",  # window type
    "nperseg": 256,
    "noverlap": 128,
    "nfft": 256,
    "padded": True,
    "axis": -1,
    "sample_rate": 384000,  # cyceles/sec
    "resolution": 20,  # ms
}


wavelet_parameters = {
    "wavelet": "haar",
    "sample_rate": 384000,
}


def get_wvt_coeffs(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the wavelet coefficients from a given audiofile.
    uses the get_audio function2
    def ssq_cwt(x,
                wavelet='gmw',
                scales='log-piecewise',
                nv=None,
                fs=None,
                t=None,
                ssq_freqs=None,
                padtype='reflect',
                squeezing='sum',
                maprange='peak',
                difftype='trig',
                difforder=None,
                gamma=None,
                vectorized=True,
                preserve_transform=None,
                astensor=True,
                order=0,
                nan_checks=None,
                patience=0,
                flipud=True,
                cache_wavelet=None,
                get_w=False,
                get_dWx=False):

     Returns:
        Tx: np.ndarray [nf x n]
            Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        Wx: np.ndarray [na x n]
            Continuous Wavelet Transform of `x`, L1-normed (see `cwt`).
        ssq_freqs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.
        scales: np.ndarray [na]
            Scales associated with rows of `Wx`.
        w: np.ndarray [na x n]  (if `get_w=True`)
            Phase transform for each element of `Wx`.
        dWx: [na x n] np.ndarray (if `get_dWx=True`)
            See `help(_cwt.cwt)`.

    """
    assert isinstance(fname, str), "Must be of type string"

    raw_buffer, sample_rate = get_audio(fname)

    Tx, Wx, *_ = ssq_cwt(x=raw_buffer, wavelet="haar", fs=sample_rate)

    return Tx, Wx


def get_stft_from_file(
    fname, type: str = "wav"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produces a tuple of information from an input audio path
    Handles both flacs and wav files

    returns:
    frequencies:  an numpy array of frequencies from the audio file
    time_seg: a numpy array of time segments from the audiofile
    Zxx:  a complex array of the stft coeffcieints from the input data

    """

    assert isinstance(fname, str), "Must be of type string"

    if type == "wav":
        raw_buffer, sample_rate = get_wav_audio(fname)

    elif type == "flac":
        raw_buffer, sample_rate = get_flac_audio(fname)

    else:
        raise TypeError

    frequencies, time_seg, Zxx = stft(
        x=raw_buffer,
        fs=sample_rate,
        window="hann",
        nperseg=stft_parameters["nperseg"],
        noverlap=stft_parameters["noverlap"],
        nfft=stft_parameters["nfft"],
        padded=stft_parameters["padded"],
        axis=stft_parameters["axis"],
    )
    return frequencies, time_seg, Zxx


def get_wav_audio(filename: str) -> Tuple[np.ndarray, int]:
    sample_rate, raw_buffer = wavfile.read(filename=filename)
    return raw_buffer, sample_rate


def get_flac_audio(filename) -> Tuple[np.ndarray, int]:
    decoder = FileDecoder(input_file=filename)
    raw_buffer, sample_rate = decoder.process()
    return raw_buffer, sample_rate


def get_audio(filename) -> Tuple[np.ndarray, int]:
    """
    loads audio given filename
    checks for extension.
    """
    fname = pathlib.Path(filename)
    parent = fname.parent
    name = fname.name
    ext = fname.suffix
    ext = str(ext).lower()

    if ext == ".wav":
        os.chdir(parent)
        print(f"{name} found with ext {ext}")
        buffer, sr = get_wav_audio(name)
    elif ext == ".flac":
        os.chdir(parent)
        print(f"{name} found with ext {ext}")
        buffer, sr = get_flac_audio(name)
    else:
        raise NotImplementedError

    return buffer, sr


if __name__ == "__main__":

    test = "p225_001.wav"
    test = pathlib.Path(test)
    test2 = (
        test.absolute().parents[3] / "Desktop" / "RAT_Filter" / "vctk_corpus" / "wav48"
    )
    test3 = test2 / "p255" / "p255_001.wav"

    get_wav_audio(test.as_posix())
    get_audio(test.as_posix())
