import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from dense import Dense

nnfs.init()

weights = np.array([[0.2, 0.5, -0.26],
                    [0.8, -0.91, -0.27],
                    [-0.5, 0.26, 0.17],
                    [1.0, -0.5, 0.87]])
biases = np.array([2, 3, 0.5])

inputs = np.array([1.0, 2.0, 3.0, 2.5])

dense = Dense(weights, biases)
outputs = dense.forward(inputs)

expected_outputs = np.array([4.8, 1.21, 2.385])

print(outputs, expected_outputs)

random_dense_1 = Dense()
print(random_dense_1.forward(inputs))

random_dense_2 = Dense()
print(random_dense_2.forward(inputs))

'''
# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])
'''
