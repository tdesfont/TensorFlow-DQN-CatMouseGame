#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verbose demo biocells_model

@author: tdesfont
"""

from biocells.biocells_model import BioCells
import numpy as np

simulation = BioCells(verbose=1)

iterate = 0
while not simulation.check_game_is_over() and iterate < 100:
    iterate += 1
    action = np.random.randint(4)
    simulation.play(action)
