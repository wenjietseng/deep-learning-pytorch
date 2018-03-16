import numpy as np
import random


class neural_network:
    def __init__(self):
        pass



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([0, 1, 1, 0])
