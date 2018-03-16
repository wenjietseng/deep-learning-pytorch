import numpy as np
import random

class neural_network:
    def __init__(self, X, Y):
        pass
    def forward_propagation(self):
        pass
    def backward_propagation(self):
        pass
    def test(self):
        pass
    def train_network(self):
        pass


# End of class

def randomized_matrix(matrix, a, b):
    pass

# Sigmoid function definition
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Derivation of sigmoid function
def dsigmoid(fx):
    return (1.0 - fx) * fx

def main():
    # X is the input matrix and Y is the output vector of XOR gate. 
    X = np.array(
        [[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]]
        )
    Y = np.array([0, 1, 1, 0])
    nn = neural_network(X, Y)
    # train neural network here

if __name__ == "__main__":
    main()