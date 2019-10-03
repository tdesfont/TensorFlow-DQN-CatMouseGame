#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is testing the discretization of the continuous state-action space
As a reminder, the state-action space is continuous in the state, but not in
action for our action as we gave it only four actions to play.

As a result of our discretization, we will be able to give as an input the
grid and not the image which is a much more compact representation. Even if
the two are very sparse.

@author: tdesfont
"""

from biocells.biocells_model import BioCells
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from discretize_screen.discretization import convert_to_grid, get_full_indicator_matrix

###############################################################################
# Test 1:

simulation = BioCells(verbose=0)

iterate = 0
while not simulation.check_game_is_over() and iterate < 100:

    iterate += 1
    action = np.random.randint(4)
    simulation.play(action)

    # Collect positions
    print('_______________________________')
    print('Position prey:    ', simulation.position_prey)
    print('Position predator:', simulation.position_predator)
    print('Check game is over:', simulation.check_game_is_over())

    # Convert to array of 2
    position_prey = simulation.position_prey[0]
    position_predator = simulation.position_predator[0]

    # Store the visualisation of the grid of shape (h, w, 1)
    fig = plt.figure()
    plt.imshow(get_full_indicator_matrix(position_prey, position_predator, dimension_option=1, bins_number=10))
    plt.savefig('test_discretization/grid_1/grid1_step_{}.jpg'.format(iterate))

    # Store the visualisation of the grid of shape (h, w, 3)
    fig = plt.figure()
    plt.imshow(get_full_indicator_matrix(position_prey, position_predator, dimension_option=3, bins_number=10));
    plt.savefig('test_discretization/grid_3/grid3_step_{}.jpg'.format(iterate))

###############################################################################
# Test 2:

simulation = BioCells(verbose=0)
position_prey = simulation.position_prey[0]
position_predator = simulation.position_predator[0]

current_screen = get_full_indicator_matrix(position_prey,
                                           position_predator,
                                           dimension_option=3,
                                           bins_number=10)

iterate = 0

while not simulation.check_game_is_over() and iterate < 100:

    previous_screen = current_screen

    iterate += 1
    action = np.random.randint(4)
    simulation.play(action)

    # Convert to array of 2
    position_prey = simulation.position_prey[0]
    position_predator = simulation.position_predator[0]

    # Store the visualisation of the grid of shape (h, w, 1)
    current_screen = get_full_indicator_matrix(position_prey,
                                               position_predator,
                                               dimension_option=3,
                                               bins_number=10)

    differential_screen = 0.6*current_screen + 0.4*previous_screen

    plt.figure()
    plt.imshow(differential_screen)
    plt.savefig('test_discretization/differential_grid_3/step_{}.jpg'.format(iterate))
