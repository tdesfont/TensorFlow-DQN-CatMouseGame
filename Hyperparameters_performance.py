#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do performance comparison for the hyperparameters
"""

import time
from tensorflow_model.policy_gradient import train, test
from tensorflow_model.deep_network.model import FullyConnected
from biocells.biocells_model import BioCells
import matplotlib.pyplot as plt
import os

assert "results" in os.listdir()

# One sample training

grid_size = 10
simulation = BioCells()
n_input = grid_size**2
model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)

start_time = time.time()
avg_lifetime, avg_reward, computation_time = train(model, simulation,
                                                   batch_size=128,
                                                   n_iterations=2,
                                                   gamma=0.7,
                                                   learning_rate=0.01,
                                                   grid_size=10,
                                                   tree_search=True,
                                                   ts_prop=0.5)
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time-start_time))
print('Reward:', test(model, simulation, grid_size=10))


"""
    1. NON RANDOM INITILISATION
"""

total_games = 600

"""
        1.1 Parameter: BATCH SIZE
"""

overall_avg_lifetime = []
overall_cost = []
overall_computational_time = []

#batch_size_range = [8, 16, 32, 64, 128, 256]
parameter_range = [10, 50, 100, 500]

for batch in parameter_range:

    n_iter = total_games / batch
    grid_size = 10
    simulation = BioCells(random_corner_init=True)
    n_input = grid_size**2
    model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)

    start_time = time.time()
    avg_lifetime, avg_reward, computation_time = train(model, simulation,
                                                       batch_size=batch,
                                                       n_iterations=int(n_iter),
                                                       gamma=0.7,
                                                       learning_rate=0.01,
                                                       grid_size=10,
                                                       tree_search=False,
                                                       ts_prop=0.5)

    end_time = time.time()

    overall_avg_lifetime.append(avg_lifetime)
    overall_cost.append(avg_reward)
    overall_computational_time.append(computation_time)

print('Done...')

plt.figure(figsize=(15,5))
plt.title('Evolution of the lifetime with respect to the bacth size (constant number of simulated games)')
for index, batch in enumerate(batch_sizes):
    batch_x = [(i+1)*batch for i in range(int(total_games/batch))]
    plt.plot(batch_x, overall_avg_lifetime[index], label='batch: {}'.format(batch))
plt.ylabel('Lifetime')
plt.xlabel('number of simulated games')
plt.grid()
plt.legend()
plt.savefig('results/corner_init/batch_size.png')
plt.show()

"""
    1.2 GAMMA
"""

overall_avg_lifetime = []
overall_cost = []
overall_computational_time = []

#batch_size_range = [8, 16, 32, 64, 128, 256]
total_games = 5000
batch_size = 128
n_iter = int(total_games/batch_size)

gamma_range = [0.2, 0.7, 0.9]

for gamma in gamma_range:

    grid_size = 10
    simulation = BioCells(random_corner_init=True)
    n_input = grid_size**2
    model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)

    start_time = time.time()
    avg_lifetime, avg_reward, computation_time = train(model, simulation,
                                                       batch_size=batch_size,
                                                       n_iterations=n_iter,
                                                       gamma=gamma,
                                                       learning_rate=0.01,
                                                       grid_size=10,
                                                       tree_search=False,
                                                       ts_prop=0.5)

    end_time = time.time()

    overall_avg_lifetime.append(avg_lifetime)
    overall_cost.append(avg_reward)
    overall_computational_time.append(computation_time)

print('Done...')

plt.figure(figsize=(15,5))
plt.title('Evolution of the lifetime with respect to the bacth size (constant number of simulated games)')
for index, batch in enumerate(batch_sizes):
    batch_x = [(i+1)*batch for i in range(int(total_games/batch))]
    plt.plot(batch_x, overall_avg_lifetime[index], label='batch: {}'.format(batch))
plt.ylabel('Lifetime')
plt.xlabel('number of simulated games')
plt.grid()
plt.legend()
plt.savefig('results/corner_init/gamma.png')
plt.show()

"""
    1.3 LEARNING RATE
"""

overall_avg_lifetime = []
overall_cost = []
overall_computational_time = []

#batch_size_range = [8, 16, 32, 64, 128, 256]
total_games = 5000
batch_size = 128
n_iter = int(total_games/batch_size)

parameter_range = [0.1, 0.01, 0.001]

for lr in parameter_range:

    grid_size = 10
    simulation = BioCells(random_corner_init=True)
    n_input = grid_size**2
    model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)

    start_time = time.time()
    avg_lifetime, avg_reward, computation_time = train(model, simulation,
                                                       batch_size=batch_size,
                                                       n_iterations=n_iter,
                                                       gamma=0.7,
                                                       learning_rate=lr,
                                                       grid_size=10,
                                                       tree_search=False,
                                                       ts_prop=0.5)

    end_time = time.time()

    overall_avg_lifetime.append(avg_lifetime)
    overall_cost.append(avg_reward)
    overall_computational_time.append(computation_time)

print('Done...')

plt.figure(figsize=(15,5))
plt.title('Evolution of the lifetime with respect to the learning rate')
for index, param in enumerate(parameter_range):
    x_values = [(i+1)*batch for i in range(int(total_games/batch))]
    plt.plot(x_values, overall_avg_lifetime[index], label="Learning rate:{}".format(param))
plt.ylabel('Lifetime')
plt.xlabel('Learning rate')
plt.grid()
plt.legend()
plt.savefig('results/corner_init/learning_rate.png')
plt.show()

"""
    1.4 GRID SIZE
"""

