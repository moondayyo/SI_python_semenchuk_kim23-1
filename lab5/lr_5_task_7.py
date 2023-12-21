import numpy as np
import neurolab as nl
import numpy.random as rand
import pylab as pl

# Standard deviation
skv = 0.05

# Centers of clusters
center = np.array([[0.2, 0.2], [0.4, 0.4], [0.7, 0.3], [0.2, 0.5]])

# Generate random data around the centers
rand.seed(0)
rand_normal = skv * rand.randn(100, 4, 2)
input = np.array([center + r for r in rand_normal])
input.shape = (100 * 4, 2)

# Shuffle data
rand.shuffle(input)

# Create net with 2 inputs and 4 neurons
net = nl.net.newc([[0.0, 1.0], [0.0, 1.0]], 4)

# Train with rule: Conscience Winner Take All (CWTA)
error = net.train(input, epochs=200, show=20)

# Plot results
pl.title('Classification Problem')
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('Train error (default MSE)')
w = net.layers[0].np['w']
pl.subplot(212)
pl.plot(input[:, 0], input[:, 1], '.', center[:, 0], center[:, 1], 'yv', w[:, 0], w[:, 1], 'p')
pl.legend(['train samples', 'real centers', 'train centers'])
pl.show()