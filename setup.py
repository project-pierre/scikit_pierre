# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# python setup.py sdist bdist_wheel
# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="scikit_pierre",
    version="0.0.2-build16",
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
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "pandas"]
)
