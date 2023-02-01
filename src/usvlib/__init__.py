import os
import sys
import lazy_loader as lazy

from ._conversions import *
from .fourier import *
from ._annotation import *
from .filters import *
from .wavelets import *
from .inputs import *
from .outputs import *
from .preprocess import *
from .viz import *


__all__ = [
    "_conversions",
    "filters",
    "preprocess",
    "_annotation",
    "inputs",
    "outputs",
    "fourier",
    "wavelets",
    "viz",
]


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "viz": ["plot_tSNE", "plot_UMAP", "plot_hessian", "plot_specs"],
    },
)
