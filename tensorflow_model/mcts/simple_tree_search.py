#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from tensorflow_model.mcts.game_simulator import Simulator

def exploration_tree(position_prey, position_predator, simulator, depth=6):
    """
        Output sequence one of the best action in the long term
    """
    init_pos_prey = position_prey
    init_pos_pred = position_predator
    directions = [0, 1, 2, 3]
    np.random.shuffle(directions)

    best_action = None
    best_reward = float('inf')

    for action in directions:

        pos_prey = init_pos_prey
        pos_pred = init_pos_pred
        sequence_pos_prey = [init_pos_prey]
        sequence_pos_pred = [init_pos_pred]
        sequence_reward = []
        sequence_action = []

        for i in range(depth):

            pos_prey, pos_pred, reward, game_over = simulator.predict(pos_prey, pos_pred, action)

            sequence_pos_prey.append(pos_prey)
            sequence_pos_pred.append(pos_pred)
            sequence_reward.append(reward)
            sequence_action.append(action)

            if game_over: break

        reward = np.sum(sequence_reward)
        # update best action
        if reward < best_reward:
            best_action = action
            best_reward = reward
    return best_action


def generate_noisy_sequence(length, position_prey, position_predator, simulator, best_action, epsilon):

    pos_prey = position_prey
    pos_pred = position_predator

    sequence_pos_prey = [pos_prey]
    sequence_pos_pred = [pos_pred]
    sequence_reward = []
    sequence_action = []

    for i in range(length):


            # allow randomness
            if i > 3 and np.random.random() > epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action = best_action

            pos_prey, pos_pred, reward, game_over = simulator.predict(pos_prey,
                                                          pos_pred,
                                                          action)

            sequence_pos_prey.append(pos_prey)
            sequence_pos_pred.append(pos_pred)
            sequence_reward.append(reward)
            sequence_action.append(best_action)

    return sequence_pos_prey[:-1], sequence_pos_pred[:-1], sequence_reward, sequence_action


if __name__ == '__main__':
    simulator = Simulator()

    sample_pos_pred = np.array([0, 10])
    sample_pos_prey = np.array([0, 0])
    print(simulator.predict(sample_pos_prey, sample_pos_pred, 0))

    sample_pos_prey = np.array([0, 0])
    sample_pos_pred = np.array([0, 4])
    print(simulator.predict(sample_pos_prey, sample_pos_pred, 0))

    best_action = exploration_tree(sample_pos_prey, sample_pos_pred, simulator, depth=6)
    batch = generate_noisy_sequence(sample_pos_prey, sample_pos_pred,
                                    simulator, best_action, epsilon=0.75)
    print(batch)
    for i in range(4):
        print(len(batch[i]))