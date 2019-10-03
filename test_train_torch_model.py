#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the PyTorch model

Cast tensor types:
https://discuss.pytorch.org/t/how-to-cast-a-tensor-to-another-type/2713

"""

import math
import random
from collections import namedtuple
from itertools import count

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch_model.DQN import DQN
from biocells.biocells_model import BioCells
from matplotlib_vision import plt_vision
from tensorboardX import SummaryWriter

# TensorBoard Writer
writer = SummaryWriter()

# Handle GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Specific format for the buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Batch replay wrapper
class ReplayMemory(object):
    """
    The DQN will sample batch episodes to have iid observations of the
    state-actions spaces. Some variants can be implmented as prioritized replay
    buffer. Here this is simply a random sampling.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Operation on the input screen
resize = T.Compose([T.ToPILImage(),
                    T.ToTensor()])

# Simulation parameters
BATCH_SIZE = 128  # Initially set to 128 (power of 2)
GAMMA = 0.999
EPS_START = 0.9  # Epsilon-greedy handling (exponentially decreasing)
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Define policy and target net (2 distinct nets but yet similar)
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# Define the optimizer
optimizer = optim.RMSprop(policy_net.parameters())
# Define the memory handler
memory = ReplayMemory(10000)
# Count the step
steps_done = 0


def select_action(state):
    """
    Define our way to select the action based on the policy.
    Can be random, probabilistic, epsilon-greedy
    Here this is an epsilon-greedy policy where the epsilon is decreasing
    exponentially.

    Input:
        state: <torch tensor numpy RGB array> Differential screen (screen - previous_screen)
    """
    global steps_done

    sample = random.random()
    # Exploration enforcement
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # Epsilon-greedy policy
    if sample > eps_threshold:
        with torch.no_grad():
            # take max among actions
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # take random action
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def optimize_model():
    """
    Optimize the model based on the TD algorithm for Q-value
    Where we call, backward and do the gradient evaluation
    """
    # Assert memory is big enough
    if len(memory) < BATCH_SIZE:
        return
    # Sample the transitions
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Avoid error while computing
    # print('Check Types:', next_state_values.type(), reward_batch.type())
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # TensorBoardX
    writer.add_scalar('loss', loss)

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def torch_screen(position_prey, position_predator):
    """
    Retrieve the screen as an image and use it as visual input (Atari like)
    """
    screen = plt_vision.get_screen(position_prey, position_predator)
    screen = screen.transpose((2, 0, 1))
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


# Train the model

simulation = BioCells()  # Define the simulation
num_episodes = 10  # Number of episodes for training
survival_time = []  # Compute the survival time

for i_episode in tqdm(range(num_episodes)):

    simulation.reset()

    # Get state (i.e. differential screen)
    position_prey = simulation.position_prey[0]
    position_predator = simulation.position_predator[0]
    last_screen = torch_screen(position_prey, position_predator)
    current_screen = torch_screen(position_prey, position_predator)
    state = current_screen - last_screen

    for t in count():
        # Handle differential screen
        last_screen = current_screen

        # Do one step (Action/Reward/GameEvent)
        action = select_action(state)
        reward = simulation.play(action.item())
        done = simulation.check_game_is_over()
        writer.add_scalar('reward', reward)
        # Cast to torch (Computation of Q-value)
        reward = torch.tensor([reward], device=device)

        # Update positions
        position_prey = simulation.position_prey[0]
        position_predator = simulation.position_predator[0]

        # Get current screen
        current_screen = torch_screen(position_prey, position_predator)

        # Compute differential screen (ie state)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            survival_time.append(t+1)
            writer.add_scalar('survival_time', t+1)
            break

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print('Simulation complete')
