#include <chrono>
#include <iostream>
#include <string_view>
#include <Eigen/Dense>

using namespace Eigen;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;

void test_eigen_vector() {
    // different ways to initialize a vector
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

void test_eigen_matrix() {
    // different ways to initialize a matrix
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
    std::cout << "call to " << functionName << " took " << elapsedTime.count() << " microseconds" << '\n';
}

void manual_neuron() {
    // manually compute output of neuron with 4 inputs
    auto start = getStartTime();

    Vector<double, 4> inputs(1.0, 2.0, 3.0, 2.5);
    Vector<double, 4> weights(0.2, 0.8, -0.5, 1.0);
    double bias = 2.0;
    double output = inputs(0) * weights(0) + 
                    inputs(1) * weights(1) + 
                    inputs(2) * weights(2) + 
                    inputs(3) * weights(3) + 
                    bias;
    std::cout << output << '\n';

    printElapsedTime(start, "manualNeuron");
}

void manual_layer() {
    // manually compute output of layer with 4 inputs and 3 neurons
    auto start = getStartTime();

    Vector<double, 4> inputs(1.0, 2.0, 3.0, 2.5);
    Vector<double, 4> weights1(0.2, 0.8, -0.5, 1.0);
    Vector<double, 4> weights2(0.5, -0.91, 0.26, -0.5);
    Vector<double, 4> weights3(-0.26, -0.27, 0.17, 0.87);
    double bias1 = 2.0;
    double bias2 = 3.0;
    double bias3 = 0.5;
    Vector<double, 3> outputs;
    outputs(0) = inputs(0) * weights1(0) + 
                 inputs(1) * weights1(1) + 
                 inputs(2) * weights1(2) + 
                 inputs(3) * weights1(3) + 
                 bias1;
    outputs(1) = inputs(0) * weights2(0) + 
                 inputs(1) * weights2(1) + 
                 inputs(2) * weights2(2) + 
                 inputs(3) * weights2(3) + 
                 bias2;
    outputs(2) = inputs(0) * weights3(0) + 
                 inputs(1) * weights3(1) + 
                 inputs(2) * weights3(2) + 
                 inputs(3) * weights3(3) + 
                 bias3;
    std::cout << outputs << '\n';

    printElapsedTime(start, "manualLayer");
}

void for_loop_layer(){
    // compute output of layer with 4 inputs and 3 neurons using a for loop
    auto start = getStartTime();

    Vector<double, 4> inputs(1.0, 2.0, 3.0, 2.5);
    Matrix<double, 3, 4> weights {{0.2, 0.8, -0.5, 1.0},
                                  {0.5, -0.91, 0.26, -0.5},
                                  {-0.26, -0.27, 0.17, 0.87}};
    Vector<double, 3> biases(2, 3, 0.5);

    Vector<double, 3> outputs;
    for (int i = 0; i < biases.size(); i++) {
        for (int j=0; j < inputs.size(); j++) {
            outputs(i) += inputs(j) * weights(i, j);
        }
        outputs(i) += biases(i);
    }
    std::cout << outputs << '\n';

    printElapsedTime(start, "forLoopLayer");
}

void dot_product_neuron(){
    // compute output of neuron with 4 inputs using dot product
    auto start = getStartTime();

    Vector<double, 4> inputs(1.0, 2.0, 3.0, 2.5);
    Vector<double, 4> weights(0.2, 0.8, -0.5, 1.0);
    double bias = 2.0;
    double output = inputs.dot(weights) + bias;
    std::cout << output << '\n';

    printElapsedTime(start, "dotProductNeuron");
}

void dot_product_layer() {
    // compute output of layer with 4 inputs and 3 neurons using matrix multiplication/dot product
    auto start = getStartTime();

    Vector<double, 4> inputs(1.0, 2.0, 3.0, 2.5);
    Matrix<double, 3, 4> weights {{0.2, 0.8, -0.5, 1.0},
                                  {0.5, -0.91, 0.26, -0.5},
                                  {-0.26, -0.27, 0.17, 0.87}};
    Vector<double, 3> biases(2, 3, 0.5);

    Vector<double, 3> outputs = weights * inputs + biases;
    std::cout << outputs << '\n';

    printElapsedTime(start, "dotProductLayer");
}

void matrix_multiplication_layer() {
    // compute output of layer with batch of data with samples with 4 inputs and layer with 3 neurons using matrix multiplication
    auto start = getStartTime();
    
    Matrix<double, 3, 4> inputs {{1.0, 2.0, 3.0, 2.5},
                                 {2.0, 5.0, -1.0, 2.0},
                                 {-1.5, 2.7, 3.3, -0.8}};
    Matrix<double, 3, 4> weights {{0.2, 0.8, -0.5, 1.0},
                                  {0.5, -0.91, 0.26, -0.5},
                                  {-0.26, -0.27, 0.17, 0.87}};
    Vector<double, 3> biases(2, 3, 0.5);

    Matrix<double, 3, 3> outputs{inputs * weights.transpose()};
    outputs = outputs.rowwise() + biases.transpose();
    std::cout << outputs << '\n'; 

    printElapsedTime(start, "matrixMultiplicationLayer");           
}
 
int main()
{   /*
    todo
    - make functions take input
    - write header
    - add gtest
    - pass and return appropriately via reference, test timing
    - templatify functions
    - more tests
    */
    test_eigen_vector();
    test_eigen_matrix();
    // these functions could've been done equivalently with std::vector
    manual_neuron();
    manual_layer();
    for_loop_layer();
    // these functions actually required Eigen
    dot_product_neuron();
    dot_product_layer();
    matrix_multiplication_layer();

    return 0;
}
