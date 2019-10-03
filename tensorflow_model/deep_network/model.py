#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define the TensorFlow model that will take in argument either:
    - the input images
    - the discretized images

Borrowed with permission of Louis Martin, from the original project DeepSnake
from Louis Martin and Pierre Stock.

@author: tdesfont
"""

import tensorflow as tf


class ConvNet:
    """
    Convolutional network taking the discretized image as input

    Return:
        probability of class
    """
    def __init__(self, n_classes, n_blocks=2):
        """
        Input:
            n_blocks: <int> Number of residual blocks
        """
        self.weights = {}  # Export the weights after training
        self.n_blocks = n_blocks
        self.n_classes = n_classes

    def conv2d(self, x, name, filter_size, nb_filter, stride=1, relu=True):
        """
        Conv2d, bias and relu activation

        Input:
            x: <tensor> Layer input
            name: <string> Useful for weights export
            filter_size: <int> Size of the filter for the 2D convolution
            nb_filter: <int> Filters number
            stride: <int> Stride of the 2D convolution
            relu: <bool> ReLu activation on output
        """
        # Get the 3rd dimension of the input
        nb_input = x.get_shape().as_list()[3]
        #
        W = tf.Variable(tf.truncated_normal([filter_size, filter_size, nb_input, nb_filter], stddev=.1))
        b = tf.Variable(tf.truncated_normal([nb_filter], stddev=1.))

        self.weights['{}_w'.format(name)] = W
        self.weights['{}_b'.format(name)] = b

        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        if relu:
            x = tf.nn.relu(x)
        return x

    def dense(self, x, name, nb_input, nb_filter, relu=True):
        """
        Dense Layer (fully connected layer)

        Input:
            x
            name
            nb_input
            nb_filter
            relu

        Return:
            x: Output of the dense layer
        """
        # Initialization of W and bias b with normal weights
        W = tf.Variable(tf.truncated_normal([nb_input, nb_filter], stddev=.1))
        b = tf.Variable(tf.truncated_normal([nb_filter], stddev=.1))
        # Naming convention
        self.weights['{}_w'.format(name)] = W
        self.weights['{}_b'.format(name)] = b
        #
        x = tf.add(tf.matmul(x, W), b)

        if relu:
            x = tf.nn.relu(x)
        return x

    def maxpool2d(self, x, k=2):
        """
        Maxpooling 2D Layer

        Input:
            x: <tensor> Layer input
            k: <int> Size of pooling window
        """
        return tf.nn.maxpool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def forward(self, input_frames):
        """
        Forward pass of the network

        Input:
            input_frames: <tensor> Input frames of the game
        """
        #
        self.input_frames = input_frames
        #
        n_classes = self.n_classes

        # First convolution 2D
        conv1 = self.conv2d(x=input_frames, name='conv1', filter_size=1, nb_filter=4, stride=1, relu=True)
        # Second convolution 2D
        conv2 = self.conv2d(x=conv1, name='conv2', filter_size=3, nb_filter=8, stride=1, relu=True)
        # Fully connected layer
        nb_input = 10*10*8
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, nb_input])
        fc1 = self.dense(x=fc1, name='fc1', nb_input=nb_input, nb_filter=128, relu=True)
        # Output class prediction
        out = self.dense(x=fc1, name='fcout', nb_input=128, nb_filter=n_classes, relu=False)
        out_probs = tf.nn.softmax(out)
        # Assign class variable
        self.out_probs = out_probs

        return out_probs

class FullyConnected:

    def __init__(self, n_input, n_hidden, n_classes):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, input_frames):
        self.input_frames = input_frames
        w1 = tf.Variable(tf.truncated_normal([self.n_input, self.n_hidden], stddev = .1))
        b1 = tf.Variable(tf.zeros([1, self.n_hidden]))
        w2 = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_classes], stddev = .1))
        b2 = tf.Variable(tf.zeros([1, self.n_classes]))
        self.weights = {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}

        input_frames = tf.reshape(input_frames, shape=[-1, self.n_input])
        hidden_layer = tf.add(tf.matmul(input_frames, w1), b1)
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.add(tf.matmul(hidden_layer, w2), b2)
        out_probs = tf.nn.softmax(out_layer)
        self.out_probs = out_probs
        return out_probs

