#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:33:58 2018

@author: tdesfont
"""

import numpy as np
from biocells.biocells_model import BioCells
from discretize_screen.discretization import get_full_indicator_matrix
import matplotlib.pyplot as plt


rewards = []
actions = []
frames = []

simulation = BioCells(verbose=0)
position_prey = simulation.position_prey[0]
position_predator = simulation.position_predator[0]

current_screen = get_full_indicator_matrix(position_prey,
                                           position_predator,
                                           dimension_option=3,
                                           bins_number=20)

iterate = 0

while not simulation.check_game_is_over() and iterate < 5:

    previous_screen = current_screen

    iterate += 1
    action = np.random.randint(4)
    simulation.play(action)

    # Convert to array of 2
    position_prey = simulation.position_prey[0]
    position_predator = simulation.position_predator[0]

    # Store the visualisation of the grid of shape (h, w, 1)
    current_screen = get_full_indicator_matrix(position_prey, position_predator, dimension_option=3, bins_number=20)
    # Compute the differential screen on 2 consecutive step of the game
    differential_screen = 0.6*current_screen + 0.4*previous_screen
    # Current frame is the input of the Neural Network
    current_frame = np.expand_dims(differential_screen, 0)

    frames.append(current_frame)

print('Game is over:', simulation.check_game_is_over())
