#include "Neuron.hpp"

#include <Eigen/Dense>

double manualNeuron(const Eigen::Vector<double, 4>& inputs, const Eigen::Vector<double, 4>& weights, double bias) {
    // manually compute output of neuron with 4 inputs
    double output = inputs(0) * weights(0) + 
                    inputs(1) * weights(1) + 
                    inputs(2) * weights(2) + 
                    inputs(3) * weights(3) + 
                    bias;

    return output;
}

Eigen::Vector<double, 3> manualLayer(const Eigen::Vector<double, 4>& inputs, const Eigen::Vector<double, 4>& weights1, const Eigen::Vector<double, 4>& weights2, const Eigen::Vector<double, 4>& weights3, double bias1, double bias2, double bias3) {
    // manually compute output of layer with 4 inputs and 3 neurons
    Eigen::Vector<double, 3> outputs;
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
