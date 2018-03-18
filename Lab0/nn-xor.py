import numpy as np
import random

class neural_network:
    def __init__(self, X, Y, epoch):
        # 2 input nodes, 3 hidden nodes, and 1 output node
        self.epoch = epoch
        self.input_node = 3
        self.hidden_node = 3
        self.output_node = 1
        
        # node weight matrix from input layer to hidden layer
        self.weight_input_hidden = np.array([0.0] * self.input_node * self.hidden_node).\
            reshape(self.input_node, self.hidden_node)

        # node weight matrix from hidden layer to output layer 
        self.weight_hidden_output = np.array([0.0] * self.hidden_node * self.output_node).\
            reshape(self.hidden_node, self.output_node)

        # activation matrices for input, hidden, and output. Initialized as 1.0
        self.activation_input = np.array([1.0] * self.input_node)
        self.activation_hidden = np.array([1.0] * self.hidden_node)
        self.activation_output = np.array([1.0] * self.output_node)

        # node weights are randomly assigned, with some bounds on values
        randomized_matrix(self.weight_input_hidden, -0.2, 0.2)
        randomized_matrix(self.weight_hidden_output, -2.0, 2.0)

        # To incorporate momentum factor, introduce another array for the 'previous change'
        self.change_input_hidden = np.array([0.0] * self.input_node * self.hidden_node).reshape(self.input_node, self.hidden_node)
        self.change_hidden_output = np.array([0.0] * self.hidden_node * self.output_node).reshape(self.hidden_node, self.output_node)

    def forward_propagation(self, feed):
        if (len(feed) != (self.input_node - 1)):
            print('Error in number of input values.')

        # activate input layer        
        for i in range(self.input_node - 1):
            self.activation_input[i] = feed[i]

        # input layer to hidden layer
        for j in range(self.hidden_node):
            self.activation_hidden[j] = sigmoid(np.sum(self.weight_input_hidden[j].T * self.activation_input))

        # hidden layer to output layer
        for k in range(self.output_node):
            self.activation_output[k] = sigmoid(np.sum(self.weight_hidden_output.T * self.activation_hidden))
            # note: weight_hidden_output[k] only takes one element from weight matrix
            
        return self.activation_output

    def backward_propagation(self, inputs, expected, output, N=0.5, M=0.1):
        # backpropagation from output layer to hidden layer
        output_deltas = (expected - output) * dsigmoid(self.activation_output)
        for j in range(self.hidden_node):
            delta_weight = self.activation_hidden[j] * output_deltas
            self.weight_hidden_output[j] += M * self.change_hidden_output[j] + N * delta_weight.reshape(1,)
            self.change_hidden_output[j] = delta_weight

        # backpropagation from hidden layer to input layer
        error_h = np.array(self.weight_hidden_output * output_deltas) # compute error
        hidden_deltas = error_h.T * dsigmoid(self.activation_hidden) 
        
        # update weights of input layer to hidden layer
        delta_weight_ih = hidden_deltas.T * self.activation_input
        self.weight_input_hidden += M * self.change_input_hidden + N * delta_weight_ih
        self.change_input_hidden = delta_weight_ih       

    def test(self, X, Y):
        # Main testing function.
        print('Train neural network with %d epochs' % self.epoch)
        for x, y in zip(X, Y):
            print("Input:", x, ' Output:', self.forward_propagation(x), '\tTarget: ', y)

    def train_network(self, X, Y):
        # Run the network for every set of input values, get the output and do backpropagation
        for i in range(self.epoch):
            for x, y in zip(X, Y):
                # do forward propagation and backpropagation
                out = np.array([self.forward_propagation(x)])
                y = np.array([y])
                self.backward_propagation(x, y, out)
        self.test(X, Y)

# End of class

def randomized_matrix(matrix, lb, ub):
    # assigned a randomized value from an uniform distribution
    # between lower and upper bound for each element
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            matrix[row][col] = random.uniform(lb, ub)

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
    epochs = 50000
    nn = neural_network(X, Y, epochs)
    nn.train_network(X, Y)

if __name__ == "__main__":
    main()