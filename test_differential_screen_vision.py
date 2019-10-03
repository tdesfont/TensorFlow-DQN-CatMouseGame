#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do a demo of biocells with display and rendering

@author: tdesfont
"""

import numpy as np
import matplotlib.pyplot as plt
from biocells.biocells_model import BioCells
from matplotlib_vision import plt_vision
import os

# Test on sample coordinates
position_prey = np.array([0, 0])
position_predator = np.array([10, 0])
screen = plt_vision.get_screen(position_prey, position_predator)
print('Screen shape:', screen.shape)

# Test on the simulations
Screens = []
Differential_screens = []

simulation = BioCells()
iterate = 0
position_prey = simulation.position_prey  # need to take [0]
position_predator = simulation.position_predator  # need to take [0]
# Store screen
current_screen = plt_vision.get_screen(position_prey[0], position_predator[0])

while not simulation.check_game_is_over() and iterate < 100:
    previous_screen = current_screen.copy()
    # Update the iterates
    iterate += 1
    # Choose random action
    action = np.random.randint(4)
    # Play the chosen action
    simulation.play(action)
    # Get position of prey/predator
    position_prey = simulation.position_prey  # need to take [0]
    position_predator = simulation.position_predator  # need to take [0]

    # Store screen
    current_screen = plt_vision.get_screen(position_prey[0], position_predator[0])
    Screens.append(current_screen)
    Differential_screens.append(current_screen-previous_screen)

# Store screens
for index in range(len(Screens)):
    plt.imshow(Screens[index]);
    plt.savefig("test_matplotlib_vision/screen_vision/screen_{}.jpg".format(index))

# Store differential screens
for index in range(len(Differential_screens)):
    plt.imshow(Differential_screens[index]);
    plt.savefig("test_matplotlib_vision/differential_screen_vision/diff_screen_{}.jpg".format(index))
