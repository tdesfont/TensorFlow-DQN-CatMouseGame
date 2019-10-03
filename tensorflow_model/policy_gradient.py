#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and test the model
"""

import time
import numpy as np
import tensorflow as tf
import pickle as pkl
from tensorflow_model.tools import discount_rewards, play_game
from tqdm import tqdm
import os

def train(model, simulation, warm_restart=False, batch_size=256,
          n_iterations=20, gamma=0.9, learning_rate=0.01, grid_size=10,
          tree_search=False, ts_prop=0.5):
    """
    Train the model by doing many simulation and computing the gradient with
    the DQN.

    Input:
        model: <class ConvolutionalNeuralNetwork>
        simulation: <class Biocells>
        warm_restart: <bool> Start with pretrained weights or not
        batch_size: <int> Size of the batch used for training
        n_iterations: <int> Number of iteration in the loss optimization
        gamma: <float> Discount factor
        learning_rate: <float> Learning rate
    """

    print('Start training...')
    start_time = time.time()

    ###########################################################################
    # Definition of the TensorFlow computation graph

    # Inputs of the model #####################################################
    # must be changed like for example simulation.discretization_size
    input_frames = tf.placeholder(tf.float32, [None, grid_size, grid_size])
    # One-hot encoded action
    # could be changed 4 -> model.n_classes which is 4 by default
    # for the four possible actions
    y_played = tf.placeholder(tf.float32, [None, 4])
    # Advantages
    advantages = tf.placeholder(tf.float32, [1, None])

    # Computation graph for loss and optimizer ################################
    # Load model
    print('Creating %s model' % model.__class__.__name__)

    # Forward pass to compute the output probabilities
    out_probs = model.forward(input_frames)
    # Loss definition and optimizer
    epsilon = 1e-15
    # Convert proba to logs
    log_probs = tf.log(tf.add(out_probs, epsilon))
    # Define loss
    loss = tf.reduce_sum(tf.matmul(advantages, (tf.multiply(y_played, log_probs))))

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss)

    # Initialize the variables
    init = tf.global_variables_initializer()

    ###########################################################################

    ops = []
    if warm_restart:
        # Load previous weights
        model_path = 'tensorflow_model/weights/weights_' + model.__class__.__name__ + '.p'
        print('Loading model from ' + model_path)
        if os.path.exists(model_path):
            weights_trained = pkl.load(open(model_path, 'rb'))
            for weight_key in weights_trained:
                assert weight_key in model.weights
                assign = tf.assign(model.weights[weight_key],
                                   weights_trained[weight_key])
                ops.append(assign)
        else:
            print('Model not found at {}'.format(model_path))

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # TensorBoard
        file_writer = tf.summary.FileWriter('runs', graph=sess.graph)

        for op in ops:
            sess.run(ops)

        avg_lifetime = np.zeros(n_iterations)
        avg_reward = np.zeros(n_iterations)

        pbar = tqdm(range(n_iterations), desc='')

        for i in pbar:

            frames_stacked = []
            targets_stacked = []
            rewards_stacked = []
            lifetime = []

            for game_count in range(batch_size):
                # Play one game
                frames, actions, rewards = play_game(simulation, model, sess, grid_size, tree_search, ts_prop)
                # Stack rewards
                lifetime.append(len(rewards))
                rewards = discount_rewards(rewards, gamma)
                rewards_stacked.append(rewards)

                for f in frames:
                    frames_stacked.append(f)
                for a in actions:
                    targets_stacked.append(a)

            # tf.summary.scalar('lifetime', tf.convert_to_tensor(np.mean(lifetime)))
            # Update progress bar description
            description = 'Lifetime: {}'.format(np.mean(lifetime))
            pbar.set_description(description)

            # Stack frames, targets and rewards
            frames_stacked = np.vstack(frames_stacked)
            targets_stacked = np.vstack(targets_stacked)
            rewards_stacked = np.hstack(rewards_stacked).astype(np.float32)
            rewards_stacked = rewards_stacked.reshape(1, len(rewards_stacked))

            # Compute average lifetime and reward
            avg_lifetime[i] = np.mean(lifetime)
            avg_reward[i] = np.mean(rewards_stacked)

            # Normalize the rewards
            rewards_stacked -= np.mean(rewards_stacked)
            std = np.std(rewards_stacked)
            if std != 0:
                rewards_stacked /= std

            # Backpropagate
            sess.run(optimizer, feed_dict={input_frames: frames_stacked,
                                           y_played: targets_stacked,
                                           advantages: rewards_stacked})

            # Record time taken by simulation and optimization
            total_time = time.time() - start_time

        # Save model
        model_path = 'tensorflow_model/weights/weights_' + model.__class__.__name__ + '.p'
        print('Saving model to ' + model_path)
        pkl.dump({k: v.eval() for k, v in model.weights.items()}, open(model_path, 'wb'))

    return avg_lifetime, avg_reward, total_time

def test(model, simulation, grid_size):
    """
    Test the model
    """
    # Load model
    input_frames = tf.placeholder(tf.float32, [None, grid_size, grid_size])
    # One pass of the model (for graph compilation)
    out_probs = model.forward(input_frames)
    # Assign weights
    model_path = 'tensorflow_model/weights/weights_' + model.__class__.__name__ + '.p'
    print('Loading model from: ' + model_path)
    weights_trained = pkl.load(open(model_path, 'rb'))

    # Assign pretrained weights to the model
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
        # Loop for n games
        lifetime = []
        n = 100
        for i in range(n):
            frames, actions, rewards = play_game(simulation, model, sess, grid_size)
            lifetime.append(len(rewards))

    return np.mean(lifetime)
