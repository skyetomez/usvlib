import sys
import os
import pathlib


import numpy as np
import matplotlib.pyplot as plt

from pywt import cwt, wavedec
from copy import deepcopy

# from ssqueezepy import ssq_cwt, ssq_stft # This is continous and not discrete.

from ._type_annotations import *
from .filters import bandpassSOS, drop_back
from .inputs import get_audio


AUDIO_PATH = "/work/pi_moorman_umass_edu/rat_recordings/single"  ## should be environmental variable
# EXPORTED_AUDIO_PATH = os.environ["AUDIO"]


class scalogram_generator:
    __slots__ = ("_audio_names", "_audio_path")

    def __init__(self, _path=AUDIO_PATH) -> None:
        self._audio_path = _path
        self._audio_names = list()

    def __str__(self) -> str:
        return "scalogram generator"

    def __repr__(self) -> str:
        return "scalogram generator object"

    @property
    def audio_path(self) -> str:
        if not self._audio_path:
            return "No audio path set"
        else:
            return self._audio_path

    @audio_path.setter
    def audio_path(self, path: str):
        _path = pathlib.Path(path)
        if _path.absolute().exists():
            self._audio_path = path
            return None
        else:
            raise NotADirectoryError

    @property
    def audio_names(self) -> List[str]:  # type: ignore
        if not self._audio_names:
            self._audio_names = self._get_audio_names()
        else:
            return self._audio_names

    @audio_names.setter
    def audio_names(self, file_list: List[str]) -> None:
        self._audio_names = file_list

    def get_scalogram(self) -> None:

        self._get_audio_names()

        for _, _, files in os.walk(top=self.audio_path, topdown=True):
            for file in files:
                pass

    def discrete_file_process(self, file: str) -> None:
        duration = 10  # minutes

        name = file[:-5].replace("-", "_").replace(" ", "_").lower()
        buffer, sr = self._get_audio_buffer(file)

        length = len(buffer)

        ten_minutes = (length // sr) * 60 * duration  # seconds * minutes
        num_sections = length // ten_minutes

        for i in range(num_sections):
            start = i * ten_minutes
            end = start + ten_minutes
            mini_buffer = buffer[start:end]
            filtered_buffer = self._bandpass_audio(mini_buffer, sr)
            coeffs = self._get_haar_wvt_decom(filtered_buffer)  # haar
            title = name + " " + "discrete scalogram"
            scalogram = self._get_haar_scalogram(coeffs, title)
            self._save(scalogram, title, discrete=True)

        return None

    def continuous_file_process(self, file: str) -> None:
        lvl = 101
        wavelet = "cmor1.5-1.0"
        duration = 10  # minutes

        name = file[:-5].replace("-", "_").replace(" ", "_").lower()
        buffer, sr = self._get_audio_buffer(file)

        length = len(buffer)

        ten_minutes = (length // sr) * 60 * duration  # seconds * minutes
        num_sections = length // ten_minutes

        for i in range(num_sections):
            start = i * ten_minutes
            end = start + ten_minutes
            mini_buffer = buffer[start:end]
            filtered_buffer = self._bandpass_audio(mini_buffer, sr)
            coeffs = self._get_morl_wvt_decom(filtered_buffer, level=lvl)  # morlet
            title = name + " " + f"cwt{wavelet} with {lvl} levels"
            scalogram = self._get_morl_scalogram(coeffs, title)
            self._save(scalogram, title, discrete=False)

        return None

    def _get_audio_names(self) -> List[str]:
        """Get a list of audios in directory

        Returns:
            List: list of names of audios in directory
        """
        for _, _, files in os.walk(top=self.audio_path, topdown=True):
            self.audio_names = deepcopy(files)

        return self.audio_names

    def _get_audio_buffer(self, file: str) -> Tuple[NDArray, int]:
        """Loads audio buffer

        TODO convert to use of next method so that audios are not
        stored as a list but are passed individually to save memory.

        Args:
            file (str): audio file names from cwd

        Returns:
            NDArray: raw audio buffer of the input audio.
        """

        return get_audio(file)

    def _clean_audio(self, raw_buffer: NDArray) -> NDArray:
        buffer = drop_back(raw_buffer)
        return buffer

    def _bandpass_audio(self, raw_buffer: NDArray, sample_rate: int) -> NDArray:
        return bandpassSOS(signal=raw_buffer, sr=sample_rate)

    def _get_haar_wvt_decom(self, buffer: NDArray, level: int = 5) -> List[NDArray]:
        coeff_decomp = wavedec(
            buffer, wavelet="haar", mode="symmetric", level=level
        )  # returns approximation and detailed coefficients
        save_path = pathlib.Path(os.getenv("DISCRETE")) / title
        np.save(save_path, coeff_decomp, allow_pickle=True)
        return coeff_decomp

    def _get_morl_wvt_decom(self, buffer: NDArray, level: int = 5) -> NDArray:
        sample_period = 1 / 384000
        lvl = 101
        wavelet = "cmor1.5-1.0"

        coeffs, *_ = cwt(
            data=buffer,
            scales=list(range(1, lvl, 1)),
            wavelet=wavelet,
            sampling_period=sample_period,
            method="fft",
        )
        save_path = pathlib.Path(os.getenv("CONTINUOUS")) / title
        np.save(save_path, coeffs, allow_pickle=True)       
        
        return np.flipud(coeffs)

    def _get_scalogram(
        self, data, title, level: int = 5, save=False
    ) -> Figure:  # broken

        if isinstance(data, list):
            if not save:
                return self._get_haar_scalogram(coeffs=data, title=title)
            else:
                res = self._get_haar_scalogram(coeffs=data, title=title)
                save_path = pathlib.Path(os.getenv("DISCRETE")) / title
                np.save(save_path, res, allow_pickle=True)
                return res

        elif isinstance(data[0].dtype, "complex128"):
            if not save:
                return self._get_morl_scalogram(
                    complex_matrix=data, title=title, level=level
                )
            else:
                res = self._get_morl_scalogram(
                    complex_matrix=np.flipud(data), title=title, level=level
                )
                save_path = pathlib.Path(os.getenv("CONTINUOUS")) / title
                np.save(save_path, res, allow_pickle=True)
                return res
        else:
            raise NotImplementedError

    def _save(self, figure: Figure, title: str, discrete=False) -> None:

        # tmp_path = os.environ["SAVE_DIR"]
        if discrete:
            path = pathlib.Path(os.getenv("DISCRETE"))
        else:
            path = pathlib.Path(os.getenv("CONTINUOUS"))

        # os.chdir(path)
        ext = "jpg"
        name = title + "." + ext
        save_path = path / name

        figure.savefig(
            fname=save_path.as_posix(),
            dpi="figure",
            format=ext,
            pad_inches=0.1,
            facecolor="auto",
            edgecolor="auto",
            orientation="landscape",
        )

        return None

    def _get_morl_scalogram(
        self, complex_matrix: NDArray, title: str, level: int = 5
    ) -> Figure:
        fig = plt.figure(figsize=(3, 2), dpi=200)
        ax = fig.gca()

        im_max = np.max(np.abs(complex_matrix))
        im_min = -np.max(np.abs(complex_matrix))

        ax.imshow(
            X=np.abs(complex_matrix),
            extent=[-1, 1, 1, level + 1],
            cmap="magma",
            aspect="auto",
            vmax=im_max,
            vmin=im_min,
        )
        ax.set_title(label=title)

        plt.show()

        return fig

    def _get_haar_scalogram(self, coeffs: List[NDArray], title: str) -> Figure:

        num_coeffs = len(coeffs)
        level = int(num_coeffs - 1)
        labels = []

        fig = plt.figure(figsize=(3, 2), dpi=200)
        ax = fig.gca()

        for i, ci in enumerate(coeffs):
            ax.imshow(
                ci.reshape(1, -1),
                extent=[0, 1000, i + 0.5, i + 1.5],
                cmap="inferno",
                aspect="auto",
                interpolation="nearest",
            )
            labels.append(f"D{level -i}")

        labels.insert(0, f"A{level}")
        labels.pop()

        ax.set_ylim(
            0.5, num_coeffs + 0.5
        )  # set the y-lim to include the six horizontal images
        ax.set_title(label=title)
        # optionally relabel the y-axis (the given labeling is 1,2,3,...)
        plt.yticks(range(1, num_coeffs + 1), labels)

        plt.show()

        return fig
