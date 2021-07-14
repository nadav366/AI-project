import random
from collections import namedtuple

import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state'))


class ExperienceReplay:
    def __init__(self, e_max=10000, state_size=32):
        self._max = e_max  # maximum number of experiences
        self.transitions = []  # total experiences the Agent stored
        self.state_size = state_size

    def get_max(self):
        """return the maximum number of experiences"""
        return self._max

    def get_num(self):
        """return the current number of experiences"""
        return len(self.transitions)

    def reset(self):
        """resets the memory, deleting all previous experiences"""
        self.transitions = []

    def get_batch(self, batch_size: int):
        """randomly choose a batch of experiences for training"""
        if batch_size < self.get_num():  # We must sample with replacements
            batch = random.choices(self.transitions, k=batch_size)
        else:
            batch = random.sample(self.transitions, k=batch_size)
        batch = Transition(*zip(*batch))
        state = np.array(batch.state).astype(np.float32)
        action = np.array(batch.action)
        next_state = np.array(batch.next_state)
        return state, action, next_state

    def add(self, state, action, next_state):
        """remove the oldest experience if the memory is full"""
        if self.get_num() > self.get_max():
            del self.transitions[0]
        """add single experience"""
        self.transitions.append(Transition(state, action, next_state))
