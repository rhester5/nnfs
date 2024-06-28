# nnfs
Working through Neural Networks from Scratch in C++

## Installation 

Download Eigen (https://eigen.tuxfamily.org/dox/GettingStarted.html), unzip/untar, and copy the Eigen folder to /usr/local/include

then from project root:

    ./install.sh

## C++ Only Installation
Download Eigen (https://eigen.tuxfamily.org/dox/GettingStarted.html), unzip/untar, and copy the Eigen folder to /usr/local/include

then from project root:

    git clone https://github.com/google/googletest.git

    git submodule add -b stable ../../pybind/pybind11 extern/pybind11

    git submodule update --init

    mkdir build

    cd build

    cmake ..

    make

## Running C++

from build directory:

    ./<executable name>

## Python Only Installation
from project root:

    python -m venv .venv

    source .venv/bin/activate

    pip install -r requirements.txt

## Running Python

Depending on your OS, you may first have to build the C++ code and copy/paste the python bindings files to the project root.

from project root:

    python plot.py