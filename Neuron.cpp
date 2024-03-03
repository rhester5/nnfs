#include "Neuron.h"

#include <Eigen/Dense>

using namespace Eigen;

double manualNeuron(const Vector<double, 4>& inputs, const Vector<double, 4>& weights, double bias) {
    // manually compute output of neuron with 4 inputs
    double output = inputs(0) * weights(0) + 
                    inputs(1) * weights(1) + 
                    inputs(2) * weights(2) + 
                    inputs(3) * weights(3) + 
                    bias;

    return output;
}

Vector<double, 3> manualLayer(const Vector<double, 4>& inputs, const Vector<double, 4>& weights1, const Vector<double, 4>& weights2, const Vector<double, 4>& weights3, double bias1, double bias2, double bias3) {
    // manually compute output of layer with 4 inputs and 3 neurons
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

    return outputs;
}

Vector<double, 3> forLoopLayer(const Vector<double, 4>& inputs, const Matrix<double, 3, 4>& weights, const Vector<double, 3>& biases){
    // compute output of layer with 4 inputs and 3 neurons using a for loop
    Vector<double, 3> outputs;
    for (int i = 0; i < biases.size(); i++) {
        for (int j=0; j < inputs.size(); j++) {
            outputs(i) += inputs(j) * weights(i, j);
        }
        outputs(i) += biases(i);
    }

    return outputs;
}

double dotProductNeuron(const Vector<double, 4>& inputs, const Vector<double, 4>& weights, double bias){
    // compute output of neuron with 4 inputs using dot product
    return inputs.dot(weights) + bias;
}

Vector<double, 3> dotProductLayer(const Vector<double, 4>& inputs, const Matrix<double, 3, 4>& weights, const Vector<double, 3>& biases) {
    // compute output of layer with 4 inputs and 3 neurons using matrix multiplication/dot product
    return weights * inputs + biases;
}

Matrix<double, 3, 3> matrixMultiplicationLayer(const Matrix<double, 3, 4>& inputs, const Matrix<double, 3, 4>& weights, const Vector<double, 3>& biases) {
    // compute output of layer with batch of data with samples with 4 inputs and layer with 3 neurons using matrix multiplication
    Matrix<double, 3, 3> outputs{inputs * weights.transpose()};
    return outputs.rowwise() + biases.transpose();    
}

Matrix<double, 3, 3> matrixMultiplicationLayerByValue(Matrix<double, 3, 4> inputs, Matrix<double, 3, 4> weights, Vector<double, 3> biases) {
    // compute output of layer with batch of data with samples with 4 inputs and layer with 3 neurons using matrix multiplication
    Matrix<double, 3, 3> outputs{inputs * weights.transpose()};
    return outputs.rowwise() + biases.transpose();    
}
