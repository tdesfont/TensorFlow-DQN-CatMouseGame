#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 19:06:03 2019

@author: tdesfont
"""


import numpy as np
from biocells.biocells_model import BioCells
from discretize_screen.discretization import get_full_indicator_matrix


rewards = []
actions = []
frames = []

simulation = BioCells(verbose=0)
position_prey = simulation.position_prey[0]
position_predator = simulation.position_predator[0]

stacked_pos_prey_predator = np.vstack((simulation.position_prey[0], simulation.position_predator[0]))

simulation.reset()
while not simulation.game_over:
    print(simulation.play(1))
    print(simulation.position_predator)

from mcts.game_simulator import Simulator
simulator = Simulator()
help(simulator.predict)
simulator.predict(np.array([0, 0]), np.array([0, 3]), 3)


