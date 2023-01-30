"""
Work in progress; 
Causes slow loading of library. 
Needs to be done by lazy loading

"""
# import os
# import sys
# import pathlib


# import numpy as np
# import ssqueezepy as sqz
# from ..iohandler import get_wvt_coeffs
# import multiprocessing

# import matplotlib.pyplot as plt
# import matplotlib as mpl

# from typing import Tuple

# """
# Inputs to the program are audiofiles.
# """


# def gen_sqz_scalogram(fname: str):
#     os.environ["SSQ_PARALLEL"] = "1"
#     Tx, _ = get_wvt_coeffs(fname)

#     plt.imshow(np.abs(Tx), aspect="auto", cmap="turbo")
#     plt.show()


# def gen_unsqz_scalogram(fname: str):
#     os.environ["SSQ_PARALLEL"] = "1"
#     _, Wx = get_wvt_coeffs(fname)
#     plt.imshow(np.abs(Wx), aspect="auto", vmin=0, vmax=0.2, cmap="turbo")
#     plt.show()
