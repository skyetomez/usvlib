import os
import sys
import pathlib

import numpy as np

from typing import Tuple
from scipy.io import wavfile  # loading wavefiles as sample rate and bits(?)
import scipy.signal as dsp  # creating spectrograms

from .inputs import get_stft_from_file
from .filters import bandpassSOS
from .preprocess import get_time_bins
from ._conversions import get_time_segments

"""
All inputs in to this library are paths to audiofiles 
"""


def gen_periodogram():
    pass


def gen_spectrogram(
    fname: str, sample_rate: int = 384000, time_res: int = 20, band_pass: bool = True
) -> np.ndarray:
    """
    TODO create custom stft function because In previous iterations,
    so that one is always available with default settings.

    Can take file name but
    Looks for environmental variable to either AUDIO directory of the audios or
    WORK directory where it look for the audio file starting at the WORK root.
    If neitehr are available, it will look in the current directory.
    """
    # constants
    n_fft = int(sample_rate / time_res)
    hop_length = n_fft // 2

    audio_dir = os.environ["AUDIO"]
    work_dir = os.environ["WORK"]

    try:
        if audio_dir:
            os.chdir(audio_dir)
        elif work_dir:
            for root, _, files in os.walk(top=work_dir):
                for file in files:
                    if file == fname:
                        path = pathlib.Path(root)
                        path = path / file
                        os.chdir(path)
        else:
            path = pathlib.Path(fname)
            path = path.absolute().parent
            os.chdir(path)

        freq_segs, time_segs, cmplxfft = get_stft_from_file(
            fname
        )  # might need to change
        realfft = np.abs(cmplxfft)

        if band_pass:
            filtered = bandpassSOS(realfft, sr=sample_rate)
            power_spec = np.log(filtered)
            return power_spec
        else:
            power_spec = np.log(realfft)
            return power_spec

    except FileNotFoundError as e:
        print(
            e,
            "File cannot be found, export path variable to WORK or AUDIO of the files or move to directory with audio files.",
        )
        sys.exit()


def gen_spectrogram_local(
    fname: str, sample_rate: int = 384000, time_res: int = 20, band_pass: bool = True
) -> np.ndarray:
    """
    TODO create custom stft function because In previous iterations,
    so that one is always available with default settings.

    Can take file name but
    Looks for environmental variable to either AUDIO directory of the audios or
    WORK directory where it look for the audio file starting at the WORK root.
    If neitehr are available, it will look in the current directory.
    """
    # constants
    path = pathlib.Path(fname)
    path = path.absolute().parent
    os.chdir(path)
    print(f"CWD is {path}")

    _, _, cmplxfft = get_stft_from_file(fname)  # might need to change
    realfft = np.abs(cmplxfft)
    print("real fft computed")
    if band_pass:
        filtered = bandpassSOS(realfft, sr=sample_rate)
        power_spec = np.log(filtered)
        return power_spec
    else:
        power_spec = np.log(realfft)
        return power_spec


def sep_spec_by_time(audio, window_size: int = 20):
    """
    Division of the spectrogram into time windows

    returns:
    time sections: a numpy array of real valued spectrogram slices; these values are not log-transformed
    spec_sections:  a numpy array of their corresponding time segments for plotting sepctrogam with correct horizontal axis
    """

    if isinstance(audio, str):
        print(f"{audio} is of type str")

        freq_band, time_seg, cmplxfft = get_stft_from_file(audio)

        print(f"freq_band shape: {freq_band.shape}")
        print(f"time_band shape: {time_seg.shape}")
        print(f"cmplx_band shape: {cmplxfft.shape}")

        realfft = np.abs(cmplxfft)
        print(f"realfft shape: {realfft.shape}")
        time_bins = get_time_bins(time_seg, width=window_size)

        time_sections = list()
        spec_sections = list()

        for segment in time_bins:
            spec_sections.append(np.take(a=audio, indices=segment, axis=-1))
            time_sections.append(np.take(a=time_seg, indices=segment))

        print(f"spec_sections elem1 shape : {spec_sections[0].shape}")
        time_sections = np.array(time_sections)
        spec_sections = np.array(spec_sections)
        print(f"spec_sections shape : {spec_sections.shape}")

        return spec_sections, time_sections, freq_band

    elif isinstance(audio, np.ndarray):

        print(f"audio is of type np.ndarray")

        freq_band, time_seg = get_time_segments(audio)

        print(f"freq_band shape: {freq_band.shape}")
        print(f"time_band shape: {time_seg.shape}")

        time_bins = get_time_bins(time_seg, width=window_size)

        time_sections = list()
        spec_sections = list()

        for segment in time_bins:
            spec_sections.append(np.take(a=audio, indices=segment, axis=-1))
            time_sections.append(np.take(a=time_seg, indices=segment))

        print(f"spec_sections elem1 shape : {spec_sections[0].shape}")
        time_sections = np.array(time_sections)
        spec_sections = np.array(spec_sections)
        print(f"spec_sections shape : {spec_sections.shape}")

        return spec_sections, time_sections, freq_band

    else:
        assert "audio variable must be either a spectrogram or a path to an audio"
        return None
