#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do a demo of biocells with display and rendering

@author: tdesfont
"""

import numpy as np
import matplotlib.pyplot as plt
from biocells.biocells_model import BioCells
from display.animation_display import SubplotAnimation

simulation = BioCells(verbose=1)

iterate = 0
while not simulation.check_game_is_over() and iterate < 100:
    iterate += 1
    action = np.random.randint(4)
    simulation.play(action)
    print("Game is over:", simulation.check_game_is_over())

ani = SubplotAnimation(np.array(simulation.store_pos_prey), np.array(simulation.store_pos_predator))
plt.show()
