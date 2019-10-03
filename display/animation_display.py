#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animation display for our simulation

@author: tdesfont
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation


class SubplotAnimation(animation.TimedAnimation):

    def __init__(self, frames_agents, frames_predator):

        fig = plt.figure(figsize=(7, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        self.frames_agents = frames_agents
        self.frames_predator = frames_predator
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        self.line_agents = Line2D([], [], marker='o', markeredgecolor='k',
                                  alpha=0.8, markersize=5,
                                  linewidth=0, color='r')

        self.line_predator = Line2D([], [], marker='o', markeredgecolor='k',
                                    alpha=0.8, markersize=10,
                                    linewidth=0, color='yellow')

        self.line_canvas = Line2D([], [], linewidth=2, color='g', alpha=0.9)

        ax1.add_line(self.line_agents)
        ax1.add_line(self.line_predator)
        ax1.add_line(self.line_canvas)

        ax1.set_xlim(-50, 50)
        ax1.set_ylim(-50, 50)

        animation.TimedAnimation.__init__(self, fig, interval=100, blit=True)

    def _draw_frame(self, framedata):

        i = framedata
        head = i-1

        x_predator = self.frames_predator[head][0]
        y_predator = self.frames_predator[head][1]
        x_agents = self.frames_agents[head][0]
        y_agents = self.frames_agents[head][1]

        self.line_predator.set_data(x_predator, y_predator)
        self.line_agents.set_data(x_agents, y_agents)
        self.line_canvas.set_data([-30, 30, 30, -30, -30],
                                  [-30, -30, 30, 30, -30])

        self._drawn_artists = [self.line_agents,
                               self.line_predator,
                               self.line_canvas]

    def new_frame_seq(self):
        return iter(range(len(self.frames_agents)))

    def __init__draw(self):

        lines = [self.line_agents, self.line_predator, self.line_canvas]

        for l in lines:
            l.set_data([], [])
