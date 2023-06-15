import sys
import os
import pathlib


import numpy as np
import matplotlib.pyplot as plt

from pywt import cwt, wavedec
from copy import deepcopy

# from ssqueezepy import ssq_cwt, ssq_stft # This is continous and not discrete.

from ._type_annotations import *
from .filters import bandpassSOS, drop_back, local_median_filter, remove_empty_points
from .fourier import gen_spectrogram_local
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

            title = (
                name
                + " "
                + f"mins {i* 10} to {(i* 10) + 10}"
                + " "
                + "discrete scalogram"
            )
            mini_buffer = buffer[start:end]
            filtered_buffer = self._bandpass_audio(mini_buffer, sr)
            coeffs = self._get_haar_wvt_decom(filtered_buffer, title)  # haar
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

        ten_minutes = (length // sr * 60) * duration  # seconds * minutes
        num_sections = length // ten_minutes

        for i in range(num_sections):
            start = i * ten_minutes
            end = start + ten_minutes
            title = (
                name
                + " "
                + f"mins {i* 10} to {(i* 10) + 10}"
                + " "
                + f"cwt {wavelet} with {lvl} levels"
            )
            mini_buffer = buffer[start:end]
            filtered_buffer = self._bandpass_audio(mini_buffer, sr)
            coeffs = self._get_morl_wvt_decom(filtered_buffer, title, level=lvl)
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

    def _bandpass_audio(self, raw_buffer: NDArray, sample_rate: int) -> NDArray:
        return bandpassSOS(signal=raw_buffer, sr=sample_rate)

    def _get_haar_wvt_decom(
        self, buffer: NDArray, title: str, level: int = 5
    ) -> List[NDArray]:
        coeffs = wavedec(buffer, wavelet="haar", mode="symmetric") 
        # returns approximation and detailed coefficients
        # Need to normalize energy per level

        scaling_factor = np.arange(1, len(coeffs), 1)
        scaling_factor = np.sqrt(1 / scaling_factor)
        scaling_factor = np.hstack([np.ones(1), scaling_factor])
        
        
        for idx, scale in enumerate(coeffs):
            coeffs[idx] = np.abs(coeffs[idx] * scale) ** 2
            coeffs[idx] = 10 * np.log10(coeffs[idx], where=coeffs[idx]>0)
        
        coeffs = coeffs[::-1]

        save_path = pathlib.Path(os.getenv("DISCRETE")) / title
        # PICKLE
        np.save(
            file=save_path.as_posix(), arr=np.array(coeffs), allow_pickle=True
        )
        return coeffs

    def _get_morl_wvt_decom(
        self, buffer: NDArray, title: str, level: int = 101, save_arrays: bool = False
    ) -> NDArray:
        sample_period = 1 / 384000
        wavelet = "cmor1.5-1.0"

        coeffs, *_ = cwt(
            data=buffer,
            scales=list(range(1, level, 1)),
            wavelet=wavelet,
            sampling_period=sample_period,
            method="fft",
        )

        if save_arrays:
            save_path = pathlib.Path(os.getenv("CONTINUOUS")) / title
            # np.save(file=save_path, arr=np.array(coeffs), allow_pickle=True)

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
        ext = "png"
        name = title + "." + ext
        save_path = path / name

        figure.savefig(
            fname=save_path.as_posix(),
            dpi="figure",
            format=ext,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="auto",
            edgecolor="auto",
            orientation="landscape",
        )

        return None

    def _get_morl_scalogram(
        self,
        complex_matrix: NDArray,
        title: str,
        level: int = 5,
        plotting: bool = False,
    ) -> Figure:
        fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False, layout="tight")
        ax = fig.gca()

        ax.set_axis_off()

        if plotting:
            plt.show()

        return fig

    def _get_haar_scalogram(
        self, coeffs: List[NDArray], title: str, plotting: bool = False
    ) -> Figure:
        num_coeffs = len(coeffs)
        level = int(num_coeffs - 1)
        labels = []

        fig = plt.figure(figsize=(4, 3), dpi=200, frameon=False, layout="tight")
        ax = fig.gca()




        for i, ci in enumerate(coeffs):
            ax.imshow(
                ci.reshape(1, -1),
                extent=[0, 1000, i + 0.5, i + 1.5],
                cmap="inferno",
                aspect="auto",
                interpolation="bilinear",
            )
        # labels.append(f"D{level -i}")

        # labels.insert(0, f"A{level}")
        # labels.pop()

        ax.set_ylim(
            0.5, num_coeffs + 0.5
        )  # set the y-lim to include the six horizontal images
        # ax.set_title(label=title, fontsize=11)
        # # optionally relabel the y-axis (the given labeling is 1,2,3,...)
        # plt.yticks(range(1, num_coeffs + 1), labels)

        ax.set_axis_off()
        if plotting:
            plt.show()

        return fig


class audio_processor:
    def __init__(self) -> None:
        return None

    def process_audio(self, fname) -> Tuple[NDArray, NDArray, NDArray]:
        f, t, spec = gen_spectrogram_local(fname, roll_off=7, crits=(19_500, 80_500))
        spec[25:31, :] = -1  # needed for manual removal of noise band
        spec[0:14, :] = -1  # needed for manual removal of noise band
        dark = drop_back(np.log10(spec))
        med_filt = local_median_filter(dark, 5)
        drop, t = remove_empty_points(med_filt, t)
        return f, t, drop

    def save(
        self,
        savedir,
        fname: str,
        array: NDArray,
        time_array: NDArray,
        freq_array: NDArray,
    ) -> None:
        sd = pathlib.Path(savedir) / fname
        with open(sd, "wb") as npz:
            np.savez_compressed(
                file=npz,
                specgram=array,
                time_array=time_array,
                freq_array=freq_array,
            )
        return None
