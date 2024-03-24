#include "Neuron.h"

#include <gtest/gtest.h>

class NeuronTest : public testing::Test {
protected:
    Eigen::Vector<double, 4> m_inputs{1.0, 2.0, 3.0, 2.5};
    Eigen::Matrix<double, 3, 4> m_weights {{0.2, 0.8, -0.5, 1.0},
                                            {0.5, -0.91, 0.26, -0.5},
                                            {-0.26, -0.27, 0.17, 0.87}};
    Eigen::Vector<double, 3> m_biases{2, 3, 0.5};     
    Eigen::Vector<double, 3> m_outputs{4.8, 1.21, 2.385}; 
};

// TODO add more tests with different size inputs and matrixMultiplicationLayer

TEST_F(NeuronTest, TestDotProductLayer) {
    ASSERT_TRUE(dotProductLayer(m_inputs, m_weights, m_biases).isApprox(m_outputs));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
