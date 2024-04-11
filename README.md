# nnfs
Working through Neural Networks from Scratch in C++

## C++ Installation
from project root:
    git clone https://github.com/google/googletest.git
    git submodule add -b stable ../../pybind/pybind11 extern/pybind11
    git submodule update --init
    mkdir build
    cd build
    cmake ..
    make

## Python Installation
from project root:
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
