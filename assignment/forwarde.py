import numpy as np

# Inputs 
i1 = 0.05
i2 = 0.10

b1 = 0.5
b2 = 0.7

#Random Weights from [-0.5, 0.5]
np.random.seed(42) 

w1 = np.random.uniform(-0.5, 0.5)
w2 = np.random.uniform(-0.5, 0.5)
w3 = np.random.uniform(-0.5, 0.5)
w4 = np.random.uniform(-0.5, 0.5)
w5 = np.random.uniform(-0.5, 0.5)
w6 = np.random.uniform(-0.5, 0.5)
w7 = np.random.uniform(-0.5, 0.5)
w8 = np.random.uniform(-0.5, 0.5)

def tanh(x):
    return np.tanh(x)


# Hidden layer
net_h1 = w1*i1 + w2*i2 + b1
h1 = tanh(net_h1)

net_h2 = w3*i1 + w4*i2 + b1
h2 = tanh(net_h2)

# Output layer
net_o1 = w5*h1 + w6*h2 + b2
o1 = tanh(net_o1)

net_o2 = w7*h1 + w8*h2 + b2
o2 = tanh(net_o2)

print("Output o1 =", o1)
print("Output o2 =", o2)