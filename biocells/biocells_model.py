#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIOCELLS MODEL:


Set up the context for the agent, environment on learning and also the main
methods as step, reward and possible actions.

The environment is a square of 30 by 30. (Its left-down corner being the
origin.) The prey is the agent controlled by our algorithm and has four actions
available.

The prey has 4 actions available (each parametrized by a step):
    0: (0,  step) - Up
    1: (0, -step) - Down
    2: (step,  0) - Right
    3: (-step, 0) - Left

Note that we are in a continuous dynamic space as the predator is moving
toward the prey.

The predator always has a deterministic greedy move as it goes straight to the
prey. The speed of the predator and the prey are proportional.

As the walk of the prey is at the beginning random, we set up its speed to
2 times the speed of the predator.

Basically, the objective of the prey is to maximize its lifetime in the circle.
The prey has a simple optimal strategy, which is to do a periodic move into
the area near the borders followed indefinitely by the predator.

Our goal is to make the prey/agent learn the optimal startegy.
Our assumption is that this learning task is closely linked to an
exploration-exploitation dilemna. Indeed, a basic strategy for the prey
is to flee in the direct opposite direction of the predator meaning that it
will be stuck in the corner in the longer-term.

"""

import numpy as np
import random


def random_corner():
    """
    Return a unitary random corner among the [up, down]x[left, right]
    (Cartesian space)
    """
    random_horizontal = (random.random() > 0.5) * 2 - 1
    random_vertical = (random.random() > 0.5) * 2 - 1
    return np.array([random_horizontal, random_vertical])


class BioCells:

    def __init__(self, rewards={'nothing': -1, 'prey_eaten': +10}, verbose=0,
                 step=1, speed_ratio=3, random_corner_init=False):
        """
        Initialise the learning environment.
        Input:
            rewards: <dict> Reward given to the agent at each move.
                    Need to check in our implementation: Cost=-Reward
            verbose: <int> Do a verbose simulation (Different from a display)
            step: <int> Size of the step for the random walk (to tune)
        """

        self.verbose = verbose
        self.rewards = rewards
        self.canvas_size = 60
        self.random_corner_init = random_corner_init

        # Define the possible actions for the prey/agent
        # Parametrization by the step is important, the less the step
        # the more random the walk and the more difficult the learning task
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

        # Reset game context and environment
        self.reset()

    def reset(self):
        """
        Reset the environment
        """
        scale = self.canvas_size / 2

        if self.random_corner_init:
            # Define initial position of the prey/agent
            self.position_prey = np.zeros((1, 2))
            self.position_predator = scale*np.reshape(random_corner(), (1, 2))
        else:
            self.position_prey = scale*(np.random.random((1,2))*2 - np.ones((1,2)))
            # Define initial position of predator in one of the corner of the area
            self.position_predator = scale*(np.random.random((1,2))*2 - np.ones((1,2)))

        if self.verbose:
            print("Initial predator position:", self.position_predator)

        # Define game events
        self.prey_eaten = False
        self.game_over = False

        # Store frames for display
        self.store_pos_prey = []  # position of prey
        self.store_pos_predator = []  # position of predator

        # Reset time count
        self.t = 0

    def move(self, h_increment, v_increment):
        """
        Update positions of predator and prey.
        The agent/prey has 4 moves available [Up, Down, Left, Right]
        The predator has a deterministic greedy move at each iteration

        Input:
            h_increment : <int> Horizontal increment
            v_increment : <int> Vertical increment
                Both takes value in [-step, 0, +step]
        """
        # Update position of predator based on agent/prey's move
        direction_predator = np.angle((self.position_prey - self.position_predator) @ np.array([1, 1j]))
        self.position_predator += self.speed_predator * np.array([np.cos(direction_predator), np.sin(direction_predator)]).T
        # Enforce area constraints on the predator position
        self.position_predator = np.minimum(self.position_predator, np.ones(self.position_predator.shape) * self.canvas_size / 2)
        self.position_predator = np.maximum(self.position_predator, np.ones(self.position_predator.shape) * -self.canvas_size / 2)
        # Update position of agent/prey w.r.t. the input move
        self.position_prey += self.speed_prey * np.array([h_increment, v_increment])
        # Enforce area constraints on the prey position
        self.position_prey = np.minimum(self.position_prey, np.ones(self.position_prey.shape) * self.canvas_size/2)
        self.position_prey = np.maximum(self.position_prey, np.ones(self.position_prey.shape) * -self.canvas_size/2)
        # Store the new positions
        self.store_frame()

    def play(self, action):
        """
        Play the input action *action* and collect the reward
        Input:
            action : <int> Index of the designated action [0, 1, 2, 3]
        """
        if self.verbose:
            print('Agent action:', action)
        # Collect horizontal and vertical increment
        (h_increment, v_increment) = self.actions[action]
        # Update prey and predator positions
        self.move(h_increment, v_increment)
        # Update lifetime
        self.t += 1

        # Handle events in occurring in the game
        # Compute euclidian distance between position and prey
        pp_distance = np.sqrt((self.position_prey - self.position_predator)**2 @ np.ones((2,1)))

        if self.verbose:
            print('Distance prey-predator: {}'.format(pp_distance))

        # Update event prey_eaten (Distance is critical if lower than 3)
        # This should be parametrized
        self.prey_eaten = pp_distance < 3

        if self.prey_eaten:
            # Collect reward
            reward = self.rewards['prey_eaten']
            if self.verbose:
                print('Prey eaten')
            # Set event game_over
            self.game_over = True
        else:
            # Collect reward
            reward = self.rewards['nothing']

        return reward

    def store_frame(self):
        self.store_pos_prey.append(list(self.position_prey[0]))
        self.store_pos_predator.append(list(self.position_predator[0]))

    def check_game_is_over(self):
        return self.game_over
