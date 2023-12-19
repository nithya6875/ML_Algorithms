import numpy as np


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(1, output_size)

        

    def forward(self, X):
        # Forward propagation
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward(self, X, y, output, learning_rate):
        # Backpropagation
        error = y - output
        output_delta = error * sigmoid_derivative(output)

        error_hidden = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = error_hidden * sigmoid_derivative(self.hidden)

        # Update weights and biases
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)


# Generate synthetic dataset for binary classification
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, (100, 1))

# Initialize neural network
input_size = 2
hidden_size = 4
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training the neural network using gradient descent
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Further implementation for other optimization algorithms can be added similarly.
