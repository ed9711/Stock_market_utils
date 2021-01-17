import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_derivative(x):
    return x*(1-x)


train_input = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
train_output = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
weight = 2 * np.random.random((3, 1)) - 1

print(weight)

for i in range(20000):
    input_layer = train_input
    output = sigmoid(np.dot(input_layer, weight))

    error = train_output - output
    adjustment = error * sig_derivative(output)
    weight += np.dot(input_layer.T, adjustment)
print(weight)
print(output)