#pragma once

#include "Dense.hpp"

#include <gtest/gtest.h>

class DenseTest : public testing::Test {
protected:
    Eigen::Matrix<double, 4, 3> m_weights {{0.2, 0.5, -0.26},
                                           {0.8, -0.91, -0.27},
                                           {-0.5, 0.26, 0.17},
                                           {1.0, -0.5, 0.87}};                                        
    Eigen::Vector<double, 3> m_biases{2, 3, 0.5};     

    
    
    Eigen::Matrix<double, 1, 4> m_inputs{1.0, 2.0, 3.0, 2.5};
    Eigen::Matrix<double, 1, 3> m_outputs{4.8, 1.21, 2.385}; 
};

// TODO add more tests with different size inputs

TEST_F(DenseTest, TestDenseForward) {
    Dense testLayer(m_weights, m_biases);
    ASSERT_TRUE(testLayer.forward(m_inputs).isApprox(m_outputs));
}
