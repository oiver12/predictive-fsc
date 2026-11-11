import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "predictive_fsc._em",
            ["cpp/em.cpp"],
            cxx_std=17,
            extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
