#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train and test the DQN

"""

import time
from tensorflow_model.policy_gradient import train, test
from tensorflow_model.deep_network.model import ConvNet, FullyConnected
from biocells.biocells_model import BioCells

# Initialize the simulation engine
simulation = BioCells()
# Initialize the model
n_input = 10*10
model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)

start_time = time.time()
train(model, simulation, batch_size=128, n_iterations=50, gamma=0.7, learning_rate=0.01, grid_size=10)
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time-start_time))
print('Reward:', test(model, simulation, grid_size))
