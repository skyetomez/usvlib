import pathlib
import sys
import os
import datetime
import pickle

import numpy as np
from pickle import HIGHEST_PROTOCOL

# from .inputs import get_audio
# from ..dataprocessing.fourier import sep_spec_by_time, gen_spectrogram
# from ..filters import drop_back


def save_narray(file: str, data: np.ndarray, suffix: str) -> None:
    """
    Saves an numpy array with name as file.
    NOW:
    file is saved to numpy_array dir in data dir which is created in same dir
    as where the script is ran.

    TODO:
    File needs to be saved to fixed numpy directory location.
    """

    if "." in file:
        name = os.path.splitext(file)[0]
    name = file + "_" + suffix

    path = pathlib.Path(file)
    parent = path.absolute().parent
    save_dir = parent / "data" / "numpy_arrays"
    print(f"saving numpy array {file} to {save_dir}")
    name = save_dir / name
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        np.save(file=name, arr=data, allow_pickle=True)
        return None
    else:
        np.save(file=name, arr=data, allow_pickle=True)
        return None


def load_narray(file: str) -> np.ndarray:
    """
    Loads an numpy array from source: file
    file needs to include path to file or can
    be file if the numpy array is in the same directory

    Example:
    fpath = "example/path/to/array.npy"
    data = load_narray(file = fpath)
    """

    try:
        print(f"Attempting to open {file} from {pathlib.Path.cwd()}")
        data = np.load(file=file, mmap_mode="r")
        return data

    except OSError as e:
        print("error in loading")
        print(e)


# def to_pickle(var, fname: str, fpath: str) -> None:
#     """
#     general pickle funtion
#     """
#     try:
#         path = pathlib.Path(fpath)

#         if not path.exists():
#             os.makedirs(path)
#             os.chdir(path)
#         else:
#             os.chdir(path)

#         if "." not in fname:
#             pickle = fname + ".pickle"
#         else:
#             tmp = fname.split(".")
#             fname = tmp[0].strip()
#             pickle = fname + ".pickle"

#         with open(pickle, "wb") as dill:
#             pickle5.dump(var, dill, HIGHEST_PROTOCOL)

#         return None

#     except OSError as e:
#         print(e)
#         sys.exit()


# def save_specs(*, audio_path: str, sample_rate=384000, theta=25):
#     """
#     Routine to create cleaned points from the spectrogram of an audio file
#     """

#     os.chdir(audio_path)

#     power_spec = gen_spectrogram(audio_path)
#     now = datetime.datetime.now()

#     # specs = cl.local_median_filter(specs)
#     specs = drop_back(power_spec, theta)
#     specs = sep_spec_by_time(specs, 20)  # in ms

#     name = os.path.split(audio_path)[-1]
#     name = os.path.splitext(name)[0]

#     fn = f"{name}_{sample_rate}_{theta}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.pickle"

#     try:
#         with open(fn, "wb") as fp:
#             pickle5.dump(specs, fp, HIGHEST_PROTOCOL)
#             print("Saved Successfully")
#     except OSError as e:
#         print(e)
#         sys.exit()


# def save_result(thing, name: str, type: str) -> int:
#     """
#     Searches for environment variable WORK to save files.
#     type can be points, spec or graph.

#     points is saved as a pickle to a directory of the same name
#     spec is saved as a pickle to a directory of the same name
#     graph is saved as a jpg to a directory of the same name
#     """

#     if "." in name:
#         tmp = name.split(".")
#         name = name[0]

#     thing_types = ["points", "spec", "graph"]

#     cwd = pathlib.Path(os.environ["WORK"])

#     if type == thing_types[0]:
#         # need to be points folder
#         PATH = cwd / "points"

#         if not PATH.exists():
#             print(f"Dir {PATH} not found, creating")
#             try:
#                 os.mkdir(PATH.as_posix())
#                 print(f"creation successful.")
#             except OSError as e:
#                 print(e)
#                 sys.exit()

#         os.chdir(PATH)

#         name = name + ".pickle"

#         PATH_NAME = PATH / name

#         with open(PATH_NAME, "wb") as dill:
#             pickle5.dump(thing, dill, HIGHEST_PROTOCOL)

#         os.chdir(cwd)

#         return 0

#     if type == thing_types[1]:
#         # need to be in spectrogram folder
#         PATH = cwd / "spectrograms"

#         if not os.path.exists(PATH):
#             try:
#                 os.mkdir(PATH.as_posix())
#                 print(f"creation successful.")
#             except OSError as e:
#                 print(e)
#                 sys.exit()

#         os.chdir(PATH)

#         name = name + ".pickle"

#         PATH_NAME = PATH / name

#         with open(PATH_NAME, "wb") as dill:
#             pickle5.dump(thing, dill, HIGHEST_PROTOCOL)

#         os.chdir(cwd)

#         return 0

#     if type == thing_types[2]:
#         name = name + ".jpg"
#         PATH = cwd / "figures"

#         if not os.path.exists(PATH):
#             try:
#                 os.mkdir(PATH.as_posix())
#                 print(f"creation successful.")
#             except OSError as e:
#                 print(e)
#                 sys.exit()

#         os.chdir(PATH)

#         PATH_NAME = PATH / name

#         thing.savefig(
#             PATH_NAME,
#             dpi="figure",
#             format="jpg",
#             pad_inches=0.1,
#             facecolor="auto",
#             edgecolor="auto",
#             orientation="landscape",
#         )

#         os.chdir(cwd)

#     return 0
