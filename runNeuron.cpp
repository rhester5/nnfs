#include "Neuron.h"

#include <chrono>
#include <iostream>
#include <string_view>
#include <Eigen/Dense>

using namespace Eigen;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;

void testEigenVector() {
    // different ways to initialize a vector in Eigen
    MatrixXd m(3,1);
    m(0,0) = 1;
    m(1,0) = 2;
    m(2,0) = 3;
    std::cout << m << '\n';

    Matrix<double, 3, 1> M(1, 2, 3);
    std::cout << M << '\n';

    Matrix<double, 3, 1> mm = {1, 2, 3};
    std::cout << mm << '\n';

    VectorXd v(3);
    v(0) = 1;
    v(1) = 2;
    v(2) = 3;
    std::cout << v << '\n';

    Vector<double, 3> V(1, 2, 3);
    std::cout << V << '\n';
}

void testEigenMatrix() {
    // different ways to initialize a matrix in Eigen
    MatrixXd m(2, 3);
    m(0,0) = 1;
    m(0,1) = 2;
    m(0,2) = 3;
    m(1,0) = 4;
    m(1,1) = 5;
    m(1,2) = 6;
    std::cout << m << '\n';

    MatrixXd M {{1, 2, 3},
                {4, 5, 6}};
    std::cout << M << '\n';

    Matrix<double, 2, 3> mm {{1, 2, 3},
                             {4, 5, 6}};
    std::cout << mm << '\n';
}

auto getStartTime() {
    return high_resolution_clock::now();
}

void printElapsedTime(auto& start, std::string_view functionName) {
    auto finish = high_resolution_clock::now();
    auto elapsedTime = duration_cast<microseconds>(finish - start);
    std::cout << functionName << " took " << elapsedTime.count() << ((elapsedTime.count() == 1) ? " microsecond" : " microseconds") << '\n';
}

int main()
{   /*
    todo
    - add gtest
    ---
    - templatify dotProductLayer and matrixMultiplicationLayer since those may actually be reused
    - more tests with different matrix and vector sizes
    ---
    - formatting (configure .editorconfig, add to gitignore)
    */

    testEigenVector();
    testEigenMatrix();

    // these functions could've been done equivalently with std::vector

    Vector<double, 4> inputs(1.0, 2.0, 3.0, 2.5);
    Vector<double, 4> weights(0.2, 0.8, -0.5, 1.0);
    double bias = 2.0;

    double manualNeuronOutput{manualNeuron(inputs, weights, bias)};
    std::cout << manualNeuronOutput << '\n';

    Vector<double, 4> weights1(0.2, 0.8, -0.5, 1.0);
    Vector<double, 4> weights2(0.5, -0.91, 0.26, -0.5);
    Vector<double, 4> weights3(-0.26, -0.27, 0.17, 0.87);
    double bias1 = 2.0;
    double bias2 = 3.0;
    double bias3 = 0.5;

    Vector<double, 3> manualLayerOutput{manualLayer(inputs, weights1, weights2, weights3, bias1, bias2, bias3)};
    std::cout << manualLayerOutput << '\n';

    Matrix<double, 3, 4> weights2D {{0.2, 0.8, -0.5, 1.0},
                                  {0.5, -0.91, 0.26, -0.5},
                                  {-0.26, -0.27, 0.17, 0.87}};
    Vector<double, 3> biases(2, 3, 0.5);

    Vector<double, 3> forLoopLayerOutput{forLoopLayer(inputs, weights2D, biases)};
    std::cout << forLoopLayerOutput << '\n';

    // these functions actually required Eigen

    double dotProductNeuronOutput{dotProductNeuron(inputs, weights, bias)};
    std::cout << dotProductNeuronOutput << '\n';

    Vector<double, 3> dotProductLayerOutput{dotProductLayer(inputs, weights2D, biases)};
    std::cout << dotProductLayerOutput << '\n';


    Matrix<double, 3, 4> batch {{1.0, 2.0, 3.0, 2.5},
                                {2.0, 5.0, -1.0, 2.0},
                                {-1.5, 2.7, 3.3, -0.8}};
    Matrix<double, 3, 3> matrixMultiplicationLayerOutput{matrixMultiplicationLayer(batch, weights2D, biases)};
    std::cout << matrixMultiplicationLayerOutput << '\n';

    // test difference in runtime between pass by reference and pass by value

    auto startByRef = getStartTime();
    for (int i = 0; i < 10000; ++i) { 
        matrixMultiplicationLayer(batch, weights2D, biases);
    }
    printElapsedTime(startByRef, "Pass by reference");

    auto startByVal = getStartTime();
    for (int i = 0; i < 10000; ++i) { 
        matrixMultiplicationLayerByValue(batch, weights2D, biases);
    }
    printElapsedTime(startByVal, "Pass by value");

    // barely any difference - guess Matrix and Vector objects aren't too expensive to copy when they're this small

    // test difference in runtime between manualLayer, forLoopLayer, and dotProductLayer

    auto startManual = getStartTime();
    for (int i = 0; i < 10000; ++i) { 
        manualLayer(inputs, weights1, weights2, weights3, bias1, bias2, bias3);
    }
    printElapsedTime(startManual, "manualLayer");

    auto startForLoop = getStartTime();
    for (int i = 0; i < 10000; ++i) { 
        forLoopLayer(inputs, weights2D, biases);
    }
    printElapsedTime(startForLoop, "forLoopLayer");

    auto startDotProduct = getStartTime();
    for (int i = 0; i < 10000; ++i) { 
        dotProductLayer(inputs, weights2D, biases);
    }
    printElapsedTime(startDotProduct, "dotProductLayer");

    // manualLayer is fast but can't generalize to any number of inputs
    // forLoopLayer is general but slow
    // dotProductLayer is both

    return 0;
}
