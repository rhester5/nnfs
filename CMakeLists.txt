# https://stackoverflow.com/questions/62910867/how-to-run-tests-and-debug-google-test-project-in-vs-code

set (CMAKE_CXX_STANDARD 20)

cmake_minimum_required(VERSION 3.12) # version can be different

project(nnfs) #name of your project

enable_testing() #to  discover tests in test explorer 

add_subdirectory(googletest) # add googletest subdirectory

include_directories(googletest/include) # this is so we can #include <gtest/gtest.h>

add_executable(testNeuron testNeuron.cpp Neuron.cpp) # add this executable
add_executable(testDense testDense.cpp)
add_executable(runNeuron runNeuron.cpp Neuron.cpp)
add_executable(runDense runDense.cpp)

target_link_libraries(testNeuron PRIVATE gtest) # link google test to this executable
target_link_libraries(testDense PRIVATE gtest)

include(GoogleTest)
gtest_discover_tests(testNeuron testDense) # discovers tests by asking the compiled test executable to enumerate its tests

add_subdirectory(extern/pybind11)
include_directories(extern/pybind11/include)
pybind11_add_module(dense Dense.cpp)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# from build directory:
# cmake ..
# make
# ./<executable_name>
