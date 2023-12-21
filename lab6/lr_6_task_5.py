import numpy as np
import neurolab as nl

# Н Є М
target = [[1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1],
          [0, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           0, 1, 1, 1, 1],
          [1, 0, 0, 0, 1,
           1, 1, 0, 1, 1,
           1, 0, 1, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1], ]

chars = ['Н', 'Є', 'М']
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

# Test network
print("\nTest on defaced Н:")
test_N = np.asfarray([1, 0, 1, 0, 1,
                      0, 1, 0, 0, 1,
                      1, 1, 0, 1, 0,
                      1, 0, 1, 0, 1,
                      0, 0, 0, 0, 1])
test_N[test_N == 0] = -1
out_N = net.sim([test_N])
print((out_N[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

print("\nTest on defaced Є:")
test_E = np.asfarray([0, 1, 1, 0, 1,
                      1, 0, 0, 0, 0,
                      1, 1, 1, 1, 1,
                      1, 0, 1, 0, 0,
                      0, 1, 1, 1, 0])
test_E[test_E == 0] = -1
out_E = net.sim([test_E])
print((out_E[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))

print("\nTest on defaced М:")
test_M = np.asfarray([1, 0, 0, 0, 1,
                      1, 0, 0, 1, 1,
                      1, 0, 1, 0, 1,
                      0, 0, 0, 0, 1,
                      1, 0, 1, 0, 0])
test_M[test_M == 0] = -1
out_M = net.sim([test_M])
print((out_M[0] == target[2]).all(), 'Sim. steps', len(net.layers[0].outs))