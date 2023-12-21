import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Generate some training data
min_value = -15
max_value = 15
num_datapoints = 130
x = np.linspace(min_value, max_value, num_datapoints)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

# Create data and labels
data = x.reshape(num_datapoints, 1)
labels = y.reshape(num_datapoints, 1)

# Plot input data
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# Define a multilayer neural network with 2 hidden layers; the first hidden layer has 10 neurons and the second one has 6 neurons, and the output layer has 1 neuron
nn = nl.net.newff([[min_value, max_value]], [10, 6, 1])

# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

# Train the neural network
error_progress = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

# Execute the neural network on the training data
output = nn.sim(data)
y_pred = output.reshape(num_datapoints)

# Plot training error
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

# Plot output
x_dense = np.linspace(min_value, max_value, num_datapoints * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted')
plt.show()