#pragma once

#include <Eigen/Dense>

template <int M, int N>
class Dense {
private:
    Eigen::Matrix<double, M, N> m_weights;
    Eigen::Vector<double, N> m_biases;
public:
    Dense(const Eigen::Matrix<double, M, N>& weights, const Eigen::Vector<double, N>& biases) 
        : m_weights{weights}
        , m_biases{biases}
        {}
    // TODO constructor (random initialize)
    // TODO maybe allocate weights and biases on the heap since they could be very large
    // -> would need to allocate in this file for random initialization
    // -> would need to have constructors, setters, and forward that take pointers as input

    void setWeights(const Eigen::Matrix<double, M, N>& weights) {m_weights = weights;}
    void setBiases(const Eigen::Vector<double, N>& biases) {m_biases = biases;}

    // B is batch size
    template <int B>
    Eigen::Matrix<double, B, N> forward(const Eigen::Matrix<double, B, M>& inputs) {
        // forward pass through layer - outputs = inputs * weights + biases
        Eigen::Matrix<double, B, N> outputs{inputs * m_weights};
        return outputs.rowwise() + m_biases.transpose();
    }
};
