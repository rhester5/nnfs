#pragma once

#include <Eigen/Dense>

// M is in-dim, N is out-dim
template <int M, int N>
class Dense {
public:
    Dense(const Eigen::Matrix<double, M, N>& weights, const Eigen::Vector<double, N>& biases) 
        : m_weights{weights}
        , m_biases{biases}
        {}
    
    Dense()
        : m_weights{0.01 * Eigen::Matrix<double, M, N>::Random()}
        , m_biases{Eigen::Vector<double, N>::Zero()}
        {}

    Eigen::Matrix<double, M, N> getWeights() {return m_weights;}
    void setWeights(const Eigen::Matrix<double, M, N>& weights) {m_weights = weights;}

    Eigen::Vector<double, N> getBiases() {return m_biases;}
    void setBiases(const Eigen::Vector<double, N>& biases) {m_biases = biases;}

    // B is batch size
    template <int B>
    Eigen::Matrix<double, B, N> forward(const Eigen::Matrix<double, B, M>& inputs) {
        // forward pass through layer - outputs = inputs * weights + biases
        Eigen::Matrix<double, B, N> outputs{inputs * m_weights};
        return outputs.rowwise() + m_biases.transpose();
    }
private:
    Eigen::Matrix<double, M, N> m_weights{};
    Eigen::Vector<double, N> m_biases{};
};
