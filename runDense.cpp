#include "Dense.h"

#include <iostream>
#include <Eigen/Dense>

int main() {

    Eigen::Matrix<double, 3, 4> batch {{1.0, 2.0, 3.0, 2.5},
                                       {2.0, 5.0, -1.0, 2.0},
                                       {-1.5, 2.7, 3.3, -0.8}};

    // TODO random
    // Dense randomLayer{4, 3};
    // Eigen::Matrix<double, 3, 3> randomOutputs{randomLayer.forward(batch)};
    // std::cout << randomOutputs << '\n';

    Eigen::Matrix<double, 4, 3> weights {{0.2, 0.5, -0.26},
                                           {0.8, -0.91, -0.27},
                                           {-0.5, 0.26, 0.17},
                                           {1.0, -0.5, 0.87}};                                     
    Eigen::Vector<double, 3> biases{2, 3, 0.5};  

    Dense myLayer(weights, biases);
    Eigen::Matrix<double, 3, 3> myOutputs{myLayer.forward(batch)};
    std::cout << myOutputs << '\n';

    return 0;
}
