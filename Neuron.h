#ifndef NEURON_H
#define NEURON_H

#include <Eigen/Dense>

double manualNeuron(const Eigen::Vector<double, 4>& inputs, const Eigen::Vector<double, 4>& weights, double bias);

Eigen::Vector<double, 3> manualLayer(const Eigen::Vector<double, 4>& inputs, const Eigen::Vector<double, 4>& weights1, const Eigen::Vector<double, 4>& weights2, const Eigen::Vector<double, 4>& weights3, double bias1, double bias2, double bias3);

template <int M, int N>
Eigen::Vector<double, M> forLoopLayer(const Eigen::Vector<double, N>& inputs, const Eigen::Matrix<double, M, N>& weights, const Eigen::Vector<double, M>& biases){
    // compute output of layer with N inputs and M neurons using a for loop
    Eigen::Vector<double, M> outputs;
    for (int i = 0; i < biases.size(); i++) {
        for (int j=0; j < inputs.size(); j++) {
            outputs(i) += inputs(j) * weights(i, j);
        }
        outputs(i) += biases(i);
    }

    return outputs;
}

template <int N>
double dotProductNeuron(const Eigen::Vector<double, N>& inputs, const Eigen::Vector<double, N>& weights, double bias){
    // compute output of neuron with N inputs using dot product
    return inputs.dot(weights) + bias;
}

template <int M, int N>
Eigen::Vector<double, M> dotProductLayer(const Eigen::Vector<double, N>& inputs, const Eigen::Matrix<double, M, N>& weights, const Eigen::Vector<double, M>& biases) {
    // compute output of layer with N inputs and M neurons using matrix multiplication/dot product
    return weights * inputs + biases;
}

template <int M, int N, int S>
Eigen::Matrix<double, S, M> matrixMultiplicationLayer(const Eigen::Matrix<double, S, N>& inputs, const Eigen::Matrix<double, M, N>& weights, const Eigen::Vector<double, M>& biases) {
    // compute output of layer with batch of data with S samples with N inputs and layer with M neurons using matrix multiplication
    // pass inputs by reference
    Eigen::Matrix<double, S, M> outputs{inputs * weights.transpose()};
    return outputs.rowwise() + biases.transpose();    
}

template <int M, int N, int S>
Eigen::Matrix<double, S, M> matrixMultiplicationLayerByValue(Eigen::Matrix<double, S, N> inputs, Eigen::Matrix<double, M, N> weights, Eigen::Vector<double, M> biases) {
    // compute output of layer with batch of data with S samples with N inputs and layer with M neurons using matrix multiplication
    // pass inputs by value
    Eigen::Matrix<double, S, M> outputs{inputs * weights.transpose()};
    return outputs.rowwise() + biases.transpose();    
}


#endif
