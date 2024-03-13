"""
Setup of Scikit-Pierre
"""
# Always prefer setuptools over distutils
import os
import sys
# To use a consistent encoding
from codecs import open as codec_open

from setuptools import setup, find_packages, Extension

try:
    import numpy as np
except ImportError:
    sys.exit("Please install numpy>=1.17.3 first.")

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = "0.0.1"

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from README.md
with codec_open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with codec_open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    install_requires = [line.strip() for line in f.read().split("\n")]

cmdclass = {}

# ext = ".pyx" if USE_CYTHON else ".c"
EXT = ".py" if USE_CYTHON else ".c"

extensions = [
    Extension(
        name="scikit_pierre.tradeoff.calibration",
        sources=["scikit_pierre/tradeoff/calibration" + EXT],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="scikit_pierre.distributions.compute_distribution",
        sources=["scikit_pierre/distributions/compute_distribution" + EXT],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="scikit_pierre.metrics.evaluation",
        sources=["scikit_pierre/metrics/evaluation" + EXT],
        include_dirs=[np.get_include()]
    ),
]

EXCLUDE_FILES = [
    "scikit_pierre/__init__.py",
    "scikit_pierre/classes/__init__.py"
]


def get_ext_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)
    return paths


if USE_CYTHON:
    extensions = cythonize(
        extensions,
        # get_ext_paths('scikit_pierre', EXCLUDE_FILES),
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
        },
    )
    cmdclass.update({"build_ext": build_ext})

# This call to setup() does all the work
setup(
    name="scikit_pierre",
    version=__version__,
    description="Scikit-Pierre is a Scientific ToolKit for Post-processing Recommendations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scikit-pierre/scikit-pierre",
    author="Diego Correa da Silva",
    author_email="diegocorrea.cc@gmail.com",
    license="MIT",
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License :: 2.0',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    keywords=(
        "Calibration, Trade-Off, "
        "Collaborative Filtering, Recommender Systems"
    ),
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=cythonize(extensions),
)
