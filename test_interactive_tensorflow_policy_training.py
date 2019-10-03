#!/usr/bin/env python3
# -*- cod, Nov 12 2018, 13:43:14)
"""

Train and Test the DQN

"""

import time
import tensorflow as tf
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_model.policy_gradient import train, test
from tensorflow_model.deep_network.model import FullyConnected
from tensorflow_model.tools import sample_from_policy
from biocells.biocells_model import BioCells
from discretize_screen.discretization import get_full_indicator_matrix

from display.animation_display import SubplotAnimation

# Initialize the simulation engine
simulation = BioCells()
# Initialize the model
n_input = 10*10
model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)

for step in range(10):

    start_time = time.time()
    train(model, simulation, batch_size=100, n_iterations=1, gamma=0.9, learning_rate=0.1, grid_size=10, warm_restart=True)
    end_time = time.time()

    # Play game by the machine
    input_frames = tf.placeholder(tf.float32, [None, 10, 10])
    out_probs = model.forward(input_frames)
    # Assign weights
    model_path = 'tensorflow_model/weights/weights_' + model.__class__.__name__ + '.p'
    print('Loading model from: ' + model_path)
    weights_trained = pkl.load(open(model_path, 'rb'))

    assigns = []
    for weight_key in weights_trained:
        assert weight_key in model.weights
        assign = tf.assign(model.weights[weight_key], weights_trained[weight_key])
        assigns.append(assign)

    # Initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Load weights
        for assign in assigns:
            sess.run(assign)

        iterate = 0
        simulation.reset()
        # Initialization
        rewards = []
        actions = []
        frames = []

        # Get positions at the beginning of the game
        position_prey = simulation.position_prey[0]
        position_predator = simulation.position_predator[0]
        # Get current screen
        current_screen = get_full_indicator_matrix(position_prey, position_predator, dimension_option=1, bins_number=10)

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

            # Store the visualisation of the grid of shape (h, w, 1)
            current_screen = get_full_indicator_matrix(position_prey, position_predator, dimension_option=1, bins_number=10)
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

    ani = SubplotAnimation(np.array(simulation.store_pos_prey), np.array(simulation.store_pos_predator))
    plt.show()
