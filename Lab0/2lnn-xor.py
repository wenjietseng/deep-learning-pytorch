import numpy as np
import random

class neural_network:
    def __init__(self, X, Y, epoch):
        # 2 input nodes, 3 hidden nodes, and 1 output node
        self.epoch = epoch
        self.input_node = 3
        self.first_hidden_node = 3
        self.second_hidden_node = 3
        self.output_node = 1
        
        self.wifh = np.random.randn(self.input_node, self.first_hidden_node)
        self.wfhsh = np.random.randn(self.first_hidden_node, self.second_hidden_node)
        self.wsho = np.random.randn(self.second_hidden_node, self.output_node)

        # activation matrices for input, hidden, and output. Initialized as 1.0
        self.activation_input = np.array([1.0] * self.input_node)
        self.activation_first_hidden = np.array([1.0] * self.first_hidden_node)
        self.activation_second_hidden = np.array([1.0] * self.second_hidden_node)
        self.activation_output = np.array([1.0] * self.output_node)

        # To incorporate momentum factor, introduce another array for the 'previous change'
        self.cifh = np.array([0.0] * self.input_node * self.first_hidden_node).\
            reshape(self.input_node, self.first_hidden_node)
        self.cfhsh = np.array([0.0] * self.first_hidden_node * self.second_hidden_node).\
            reshape(self.first_hidden_node, self.second_hidden_node)
        self.csho = np.array([0.0] * self.second_hidden_node * self.output_node).\
            reshape(self.second_hidden_node, self.output_node)

    def forward_propagation(self, feed):
        if (len(feed) != (self.input_node - 1)):
            print('Error in number of input values.')

        # activate input layer        
        for i in range(self.input_node - 1):
            self.activation_input[i] = feed[i]

        # input layer to first hidden layer
        for j in range(self.first_hidden_node):
            self.activation_first_hidden[j] = sigmoid(np.sum(self.wifh[j].T * self.activation_input))

        for k in range(self.second_hidden_node):
            self.activation_second_hidden[k] = sigmoid(np.sum(self.wfhsh[k].T * self.activation_first_hidden))

        # hidden layer to output layer
        for l in range(self.output_node):
            self.activation_output[l] = sigmoid(np.sum(self.wsho.T * self.activation_second_hidden))
            
        return self.activation_output

    def backward_propagation(self, inputs, expected, output, N=0.5, M=0.1):
        # backpropagation from output layer to second hidden layer
        output_deltas_osh = (expected - output) * dsigmoid(self.activation_output)
        for k in range(self.second_hidden_node):
            delta_weight = self.activation_second_hidden[k] * output_deltas_osh
            self.wsho[k] += M * self.csho[k] + N * delta_weight.reshape(1,)
            self.csho[k] = delta_weight

        # backpropagation from second hidden layer to first hidden layer
        error_shfh = np.array(self.wsho * output_deltas_osh)
        shfh_deltas = error_shfh.T * dsigmoid(self.activation_second_hidden)

        # update weights of first hidden layer to second hidden layer
        delta_weight_shfh = shfh_deltas.T * self.activation_first_hidden
        self.wfhsh += M * self.cfhsh + N * delta_weight_shfh
        self.cfhsh = delta_weight_shfh   

        # backpropagation from first hidden layer to input layer
        error_fhi = np.array(self.wfhsh * shfh_deltas) # compute error
        fhi_deltas = error_fhi.T * dsigmoid(self.activation_first_hidden) 
        
        # update weights of input layer to hidden layer
        delta_weight_ifh = fhi_deltas.T * self.activation_input
        self.wifh += M * self.cifh + N * delta_weight_ifh
        self.cifh = delta_weight_ifh       

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
    epochs = 500
    nn = neural_network(X, Y, epochs)
    nn.train_network(X, Y)

if __name__ == "__main__":
    main()