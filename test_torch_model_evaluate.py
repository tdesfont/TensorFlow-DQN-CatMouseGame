#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Evaluate the model with the trained net

"""
from biocells.biocells_model import BioCells

# verbose one trajectory simulation
simulation=BioCells(verbose=0)

i = 0
while i < 100 and not(simulation.check_game_is_over()):
    i += 1

    position_prey = simulation.position_prey[0]
    position_predator = simulation.position_predator[0]
    last_screen = torch_screen(position_prey, position_predator)
    current_screen = torch_screen(position_prey, position_predator)
    state = current_screen - last_screen

    action = int(select_action(state=state))
    simulation.play(action)

print('GAME OVER:', simulation.check_game_is_over())
print(i)

path_predator = np.array(simulation.store_pos_predator)
path_prey = np.array(simulation.store_pos_prey)
plt.xlim([-40, 40])
plt.ylim([-40, 40])
plt.plot(path_predator[0][0], path_predator[0][1], 'ro')
plt.plot(path_predator[-1][0], path_predator[-1][1], 'ko')
plt.plot(path_predator[:, 0], path_predator[:, 1], 'r--')

plt.plot(path_prey[0][0], path_prey[0][1], 'ro')
plt.plot(path_prey[-1][0], path_prey[-1][1], 'ko')
plt.plot(path_prey[:, 0], path_prey[:, 1], 'k--')
