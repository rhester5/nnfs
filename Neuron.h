#ifndef NEURON_H
#define NEURON_H

#include <Eigen/Dense>

double manualNeuron(const Eigen::Vector<double, 4>& inputs, const Eigen::Vector<double, 4>& weights, double bias);

Eigen::Vector<double, 3> manualLayer(const Eigen::Vector<double, 4>& inputs, const Eigen::Vector<double, 4>& weights1, const Eigen::Vector<double, 4>& weights2, const Eigen::Vector<double, 4>& weights3, double bias1, double bias2, double bias3);

Eigen::Vector<double, 3> forLoopLayer(const Eigen::Vector<double, 4>& inputs, const Eigen::Matrix<double, 3, 4>& weights, const Eigen::Vector<double, 3>& biases);

double dotProductNeuron(const Eigen::Vector<double, 4>& inputs, const Eigen::Vector<double, 4>& weights, double bias);

Eigen::Vector<double, 3> dotProductLayer(const Eigen::Vector<double, 4>& inputs, const Eigen::Matrix<double, 3, 4>& weights, const Eigen::Vector<double, 3>& biases);

Eigen::Matrix<double, 3, 3> MatrixMultiplicationLayer(const Eigen::Matrix<double, 3, 4>& inputs, const Eigen::Matrix<double, 3, 4>& weights, const Eigen::Vector<double, 3>& biases);

Eigen::Matrix<double, 3, 3> MatrixMultiplicationLayerByValue(Eigen::Matrix<double, 3, 4> inputs, Eigen::Matrix<double, 3, 4> weights, Eigen::Vector<double, 3> biases);

#endif
