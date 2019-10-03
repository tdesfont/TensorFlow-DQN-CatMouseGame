#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic tools for the training and test.

@author: tdesfont
"""

import numpy as np
from numpy import random
from discretize_screen.discretization import get_full_indicator_matrix

from tensorflow_model.mcts.game_simulator import Simulator
from tensorflow_model.mcts.simple_tree_search import exploration_tree, generate_noisy_sequence

def sample_from_policy(t):
    """
    Sample from a cdf by inverse uniform
    """
    # Uniform sampling
    p = random.random()
    # Compute cdf
    cdf = np.cumsum(t)
    return np.where(cdf >= p)[0][0]


def discount_rewards(rewards, gamma):
    """
    Compute the discounted reward from the reward trace
    """
    rewards_new = np.zeros(len(rewards))
    discount_sum = 0
    for i in reversed(range(len(rewards))):
        discount_sum *= gamma
        discount_sum += rewards[i]
        rewards_new[i] = discount_sum
    return rewards_new


def play_game(simulation, model, sess, grid_size=10, tree_search=False, ts_prop=0.5):
    """
        Play one game with the biocells simulator the ConvNet model.
        Actions are chosen by the ConvNet model.

        Input:
            tree_search: Boolean add tree search
    """
    simulation.reset()
    # Initialization
    rewards = []
    actions = []
    frames = []

    memory = 5
    queue_pos_prey = []
    queue_pos_predator = []

    iterate = 0

    # Get positions at the beginning of the game
    position_prey = simulation.position_prey[0]
    position_predator = simulation.position_predator[0]
    # Insert in queue
    queue_pos_prey.insert(0, position_prey)
    queue_pos_predator.insert(0, position_predator)
    # Get current screen
    current_screen = get_full_indicator_matrix(position_prey, position_predator, dimension_option=1, bins_number=grid_size)

    # Loop on a game episode
    while not simulation.check_game_is_over() and iterate < 500:
        # Define the previous screen for differential screen computation
        previous_screen = current_screen

        iterate += 1
        action = np.random.randint(4)
        simulation.play(action)

        # Convert to array of 2
        position_prey = simulation.position_prey[0]
        position_predator = simulation.position_predator[0]

        if len(queue_pos_prey)>memory:
            queue_pos_prey.insert(0, position_prey)
            queue_pos_predator.insert(0, position_predator)
            queue_pos_prey.pop()
            queue_pos_predator.pop()
        else:
            queue_pos_prey.insert(0, position_prey)
            queue_pos_predator.insert(0, position_predator)

        # Store the visualisation of the grid of shape (h, w, 1)
        current_screen = get_full_indicator_matrix(position_prey, position_predator, dimension_option=1, bins_number=grid_size)
        # Compute the differential screen between the two
        differential_screen = 0.6*current_screen + 0.4*previous_screen
        # Convert as input for the game
        current_frame = np.expand_dims(differential_screen, 0)
        frames.append(current_frame)

        # Call TensorFlow model with current frame as input
        policy = np.ravel(sess.run(model.out_probs, feed_dict={model.input_frames: current_frame}))
        # Sample action from returned policy and convert to one-hot vector
        action = sample_from_policy(policy)
        # The target action is the proposed action by the policy
        target = np.zeros(4)
        target[action] = 1
        # Play the sampled action
        reward = simulation.play(action)
        rewards.append(reward)
        # Add one-hot encoder
        actions.append(target)

    # The simulation is now likely to be game over
    # we now come with simple tree search to go a bit further in the exploration
    bernoulli = np.random.random()
    if len(actions) > memory and tree_search and bernoulli < ts_prop:

        ante_mortem_frames = frames[:-memory]
        ante_mortem_actions = actions[:-memory]
        ante_mortem_rewards = rewards[:-memory]

        pos_prey = queue_pos_prey[-1]
        pos_predator = queue_pos_predator[-1]

        simulator = Simulator()
        best_action = exploration_tree(pos_prey, pos_predator, simulator, depth=50)
        full_batch = generate_noisy_sequence(10, position_prey, position_predator, simulator, best_action, epsilon=1)

        post_mortem_frames = []
        current_screen = get_full_indicator_matrix(pos_prey, pos_predator, dimension_option=1, bins_number=grid_size)
        for timestep in range(len(full_batch[0])):
            position_prey = full_batch[0][timestep]
            position_predator = full_batch[1][timestep]
            previous_screen = current_screen
            current_screen = get_full_indicator_matrix(position_prey, position_predator, dimension_option=1, bins_number=grid_size)
            differential_screen = 0.6*current_screen + 0.4*previous_screen
            current_frame = np.expand_dims(differential_screen, 0)
            post_mortem_frames.append(current_frame)


        post_mortem_rewards = full_batch[2]

        post_mortem_actions = []
        for action in full_batch[3]:
            target = np.zeros(4)
            target[action] = 1
            post_mortem_actions.append(target)

        frames = ante_mortem_frames + post_mortem_frames
        actions = ante_mortem_actions + post_mortem_actions
        rewards = ante_mortem_rewards + post_mortem_rewards

    return np.array(frames), np.array(actions), np.array(rewards)
