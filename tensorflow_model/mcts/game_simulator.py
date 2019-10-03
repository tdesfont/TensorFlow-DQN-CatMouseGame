#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To compute TreeSearch, the agent needs to know how the game works perfectly.
This is a strong assumption in our learning task.
"""

import numpy as np

class Simulator():
    """
        A simulator for the game, that assumes that the cell knows
        the whole game process
        Global copy of Biocells class but without local variables and
        sequential storage for the plays.
    """
    def __init__(self, rewards={'nothing': -1, 'prey_eaten': +10},
                 step=1, speed_ratio=3):
        self.rewards = rewards
        self.canvas_size = 60
        self.step = step
        self.actions = {
                0: (0, self.step),
                1: (0, -self.step),
                2: (self.step, 0),
                3: (-self.step, 0)
                }
                # Define speed of predator and prey
        self.speed_predator = 1
        self.speed_ratio = speed_ratio
        # Set the speed of the prey
        self.speed_prey = self.speed_ratio*self.speed_predator

        self.position_predator = None
        self.position_prey = None

    def predict(self, position_prey, position_predator, action):
        """
            Assuming the input position_prey and position_predator are valid

            Output:
                new position of prey
                new position of predator
                new reward
                new game event Game Over
        """
        (h_increment, v_increment) = self.actions[action]
        direction_predator = np.angle((position_prey - position_predator) @ np.array([1, 1j]))

        new_position_predator = position_predator + self.speed_predator * np.array([np.cos(direction_predator), np.sin(direction_predator)]).T
        new_position_predator = np.minimum(new_position_predator, np.ones(new_position_predator.shape) * self.canvas_size / 2)
        new_position_predator = np.maximum(new_position_predator, np.ones(new_position_predator.shape) * -self.canvas_size / 2)

        new_position_prey = position_prey + self.speed_prey * np.array([h_increment, v_increment])
        # Enforce area constraints on the prey position
        new_position_prey = np.minimum(new_position_prey, np.ones(new_position_prey.shape) * self.canvas_size/2)
        new_position_prey = np.maximum(new_position_prey, np.ones(new_position_prey.shape) * -self.canvas_size/2)

        pp_distance = np.sqrt((new_position_prey - new_position_predator)**2 @ np.ones((2,1)))
        prey_eaten = pp_distance < 3

        if prey_eaten:
            # Collect reward
            reward = self.rewards['prey_eaten']
            game_over = True
        else:
            # Collect reward
            reward = self.rewards['nothing']
            game_over = False

        return new_position_prey, new_position_predator, reward, game_over

if __name__ == '__main__':

    simulator = Simulator()

    sample_pos_pred = np.array([0, 10])
    sample_pos_prey = np.array([0, 0])
    print(simulator.predict(sample_pos_prey, sample_pos_pred, 0))

    sample_pos_prey = np.array([0, 0])
    sample_pos_pred = np.array([0, 4])
    print(simulator.predict(sample_pos_prey, sample_pos_pred, 0))