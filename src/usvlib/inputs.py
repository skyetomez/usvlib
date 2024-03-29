import os
import sys
import pathlib


import numpy as np
import soundfile as sf
import multiprocessing as mp

# from ssqueezepy import ssq_cwt

from ._type_annotations import *
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


# def get_wvt_coeffs(fname: str) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Get the wavelet coefficients from a given audiofile.
#     uses the get_audio function2
#     def ssq_cwt(x,
#                 wavelet='gmw',
#                 scales='log-piecewise',
#                 nv=None,
#                 fs=None,
#                 t=None,
#                 ssq_freqs=None,
#                 padtype='reflect',
#                 squeezing='sum',
#                 maprange='peak',
#                 difftype='trig',
#                 difforder=None,
#                 gamma=None,
#                 vectorized=True,
#                 preserve_transform=None,
#                 astensor=True,
#                 order=0,
#                 nan_checks=None,
#                 patience=0,
#                 flipud=True,
#                 cache_wavelet=None,
#                 get_w=False,
#                 get_dWx=False):

#      Returns:
#         Tx: np.ndarray [nf x n]
#             Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
#             (nf = len(ssq_freqs); n = len(x))
#             `nf = na` by default, where `na = len(scales)`.
#         Wx: np.ndarray [na x n]
#             Continuous Wavelet Transform of `x`, L1-normed (see `cwt`).
#         ssq_freqs: np.ndarray [nf]
#             Frequencies associated with rows of `Tx`.
#         scales: np.ndarray [na]
#             Scales associated with rows of `Wx`.
#         w: np.ndarray [na x n]  (if `get_w=True`)
#             Phase transform for each element of `Wx`.
#         dWx: [na x n] np.ndarray (if `get_dWx=True`)
#             See `help(_cwt.cwt)`.

#     """
#     assert isinstance(fname, str), "Must be of type string"

#     raw_buffer, sample_rate = get_audio(fname)

#     Tx, Wx, *_ = ssq_cwt(x=raw_buffer, wavelet="haar", fs=sample_rate)

#     return Tx, Wx


def get_stft_from_file(fname) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Produces a tuple of information from an input audio path
    Handles both flacs and wav files

    returns:
    frequencies:  an numpy array of frequencies from the audio file
    time_seg: a numpy array of time segments from the audiofile
    Zxx:  a complex array of the stft coeffcieints from the input data

    """

    raw_buffer, sample_rate = sf.read(fname)

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


def get_wav_audio(filename: str) -> Tuple[NDArray, int]:
    """load wav file using wavfile DEPRECEATED

    Args:
        filename (str): wavfile with .wav extension

    Returns:
        Tuple[NDArray, int]: _description_
    """
    sample_rate, raw_buffer = wavfile.read(filename=filename)
    return raw_buffer, sample_rate


def get_flac_audio(filename) -> Tuple[NDArray, int]:
    """load flac file using pyflac DEPRECEATED

    Args:
        filename (_type_): flac file with .flac extensions

    Returns:
        Tuple[np.ndarray, int]: _description_
    """
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
        buffer, sr = sf.read(file=name)
    elif ext == ".flac":
        os.chdir(parent)
        print(f"{name} found with ext {ext}")
        buffer, sr = sf.read(file=name)
    else:
        raise NotImplementedError

    return buffer, sr


def get_narray(file: str) -> NDArray:  # type: ignore
    """
    Loads an numpy array from source: file
    file needs to include path to file or can
    be file if the numpy array is in the same directory

    Example:
    fpath = "example/path/to/array.npy"
    data = load_narray(file = fpath)
    """

    print(f"Attempting to open {file} from {pathlib.Path.cwd()}")

    try:
        data = np.load(file=file, mmap_mode="r", allow_pickle=True)
        print(f"done!{file} was succesfully loaded")
        return data

    except OSError as e:
        print("error in loading")
        print(e)


def unpickle(file: str) -> NDArray:  # type: ignore
    """
    Loads an numpy array from source: file
    file needs to include path to file or can
    be file if the numpy array is in the same directory

    Example:
    fpath = "example/path/to/array.npy"
    data = load_narray(file = fpath)
    """

    print(f"Attempting to open {file} from {pathlib.Path.cwd()}")

    try:
        with open(file, mode="rb") as jar:
            data = pickle5.load(jar)
            print(f"done!{file} was succesfully loaded")
        return data

    except OSError as e:
        print("error in loading")
        print(e)


def get_cpu_count() -> int:

    if os.environ["SLURM_CPUS_PER_TASK"]:

        return int(os.environ["SLURM_CPUS_PER_TASK"])

    else:

        cpu_count = mp.cpu_count()

        if len(os.sched_getaffinity(0)) < cpu_count:
            try:
                os.sched_setaffinity(0, range(cpu_count))
            except OSError:
                print("Could not set affinity")

        n = max(len(os.sched_getaffinity(0)), 96)
        print("Using", n, "processes for the pool")
        return n


if __name__ == "__main__":

    test = "p225_001.wav"
    test = pathlib.Path(test)
    test2 = (
        test.absolute().parents[3] / "Desktop" / "RAT_Filter" / "vctk_corpus" / "wav48"
    )
    test3 = test2 / "p255" / "p255_001.wav"

    get_wav_audio(test.as_posix())
    get_audio(test.as_posix())
