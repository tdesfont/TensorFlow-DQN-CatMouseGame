#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:07:18 2018

@author: tdesfont
"""

import numpy as np
from scipy import stats


def convert_to_grid(coordinates, bins_number=15):
    """
    Convert a 2d array coordinates to a discrete grid to indicate the
    agent where the agent is.

    Here the area is in [-30, 30]x[-30, 30] but we add some padding.

    Input:
        coordinates : <np.array of length 2> Coordinates of the agent
    Return:
        statistic: <np.array> 2d histogram
    """
    statistic, _, _, binnumber = stats.binned_statistic_2d(
        np.array([coordinates[0]]),
        np.array([coordinates[1]]),
        None, 'count',
        bins=[bins_number, bins_number],
        range=[(-35, 35), (-35, 35)], expand_binnumbers=True)
    return statistic


def get_full_indicator_matrix(coord_prey, coord_predator, dimension_option=3, bins_number=15):
    """
    Assemble the two information of positions we have in a single matrix
    designed to serve as input to the neural network

    Input:
        dimension_option <int>
    """
    statistic_prey = convert_to_grid(coord_prey, bins_number=bins_number)
    shape = statistic_prey.shape
    statistic_predator = convert_to_grid(coord_predator, bins_number=bins_number)

    if dimension_option == 1:
        full_indicator_matrix = 3*statistic_prey - 2*statistic_predator
        return full_indicator_matrix

    elif dimension_option == 2:
        prey_indicator_matrix = np.zeros((shape[0], shape[1], 2))
        prey_indicator_matrix[statistic_prey == 1] = np.array([[1, 0]])

        predator_indicator_matrix = np.zeros((shape[0], shape[1], 2))
        predator_indicator_matrix[statistic_predator == 1] = np.array([[0, 1]])

        full_indicator_matrix = predator_indicator_matrix + prey_indicator_matrix
        return full_indicator_matrix

    elif dimension_option == 3:
        prey_indicator_matrix = np.zeros((shape[0], shape[1], 3))
        prey_indicator_matrix[statistic_prey == 1] = np.array([[1, 0, 0]])

        predator_indicator_matrix = np.zeros((shape[0], shape[1], 3))
        predator_indicator_matrix[statistic_predator == 1] = np.array([[0, 0, 1]])

        full_indicator_matrix = predator_indicator_matrix + prey_indicator_matrix
        return full_indicator_matrix


