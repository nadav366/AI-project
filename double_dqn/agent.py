import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import clone_model
from tqdm import tqdm
from multiprocessing import Process

from evaluation_utils import fights

from double_dqn.experience_replay import ExperienceReplay


class DQNAgent:
    def __init__(self, env, model: tf.keras.models.Model,
                 net_update_rate: int = 25,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 1e-6,
                 discount: float = 0.95,
                 punishment: float = 0.0,
                 batch_size: int = 128):
        # set hyper parameters
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.net_updating_rate = net_update_rate
        self.punishment = punishment
        self.batch_size = batch_size

        # set environment
        self.env = env
        self.state_shape = tuple(model.input.shape.as_list()[1:])
        self.action_shape = tuple(model.output.shape.as_list()[1:])

        # the number of experience per batch for batch learning
        # Experience Replay for batch learning
        self.exp_rep = ExperienceReplay(state_size=self.state_shape[0])

        # Deep Q Network
        self.discount = discount
        self.q_net = None
        self.target_net = None
        self.set_model(model)

    def get_action(self, state: np.ndarray, eps) -> int:
        """Given a state returns a random action with probability eps, and argmax(q_net(state)) with probability 1-eps.
           (only legal actions are considered)"""

        if np.random.random() >= eps:  # Exploitation
            values = self.q_net.predict(state[np.newaxis, ...])
            # Calculate the Q-value of each action
            # probs = tf.math.softmax(q).numpy().flatten()
            # choice = np.random.choice(a=[0, 1, 2], p=probs)
            choice = np.random.choice(np.flatnonzero(values == np.max(values)))
            return choice
        return np.random.choice(np.arange(3))

    def update_net(self):
        """ if there are more than batch_size experiences, Optimizes the network's weights using the Double-Q-learning
         algorithm with a batch of experiences, else returns"""
        if self.exp_rep.get_num() < self.batch_size:
            return
        batch = self.exp_rep.get_batch(self.batch_size)
        self.fit(*batch)

    def train(self, episodes: int, train_dir, step_name,
              max_actions: int = None, step_index=-1,
              checkpoint_rate=300, exploration_rate=None, state_size=32):
        """
        Runs a training session for the agent
        :param episodes: number of episodes to train.
        :param max_actions: max number of steps in an episode. if 0, each episode runs until reaching a terminal state.
        :param batch_size: number of experiences to learn from in each net_update.
        :return A tuple containing 2 lists. the first is a list of accumulated rewards for each training episode.
                The second list contains the number of actions taken in each training episode.
        """
        # set hyper parameters
        exploration_rate = self.exploration_rate if exploration_rate is None else exploration_rate
        num_actions = []
        # start training
        for i in tqdm(range(episodes)):
            self.env.reset()  # Reset the environment for a new episode
            state = self.env.get_state(state_size=state_size)  # Get starting state
            step = 0
            while max_actions is None or step <= max_actions:
                step += 1
                action = self.get_action(state, exploration_rate)
                next_state, reward = self.env.step(action, state_size=state_size)
                # Add experience to memory
                self.exp_rep.add(state, action, next_state)
                self.update_net()  # Optimize the DoubleQ-net
                if next_state is None:  # The action taken led to a  terminal state
                    break
                if (step % self.net_updating_rate) == 0:
                    # update target network
                    self.align_target_model()

                state = next_state

            # Update total_rewards and num_actions to keep track of progress
            num_actions.append(step)
            # Update target network at the end of the episode
            self.align_target_model()  # Optimize the DoubleQ-net
            if (step % self.net_updating_rate) == 0:
                # update target network
                self.align_target_model()

            if self.exp_rep.get_num() > self.batch_size:
                if (i % checkpoint_rate) == checkpoint_rate - 1:
                    save_path = self.save_data_of_cp(num_actions, step_name, train_dir, i)
            with open(os.path.join(train_dir, 'exploration_rate.txt'), 'a') as exp_rate:
                exp_rate.writelines([str(exploration_rate) + '\n'])

            # Update exploration rate
            exploration_rate = max(0.1, 0.01 + (exploration_rate - 0.01) * np.exp(-self.exploration_decay * (i + 1)))
        self.exploration_rate = exploration_rate
        return num_actions, exploration_rate

    def save_data_of_cp(self, num_actions, step_name, train_dir, i):
        save_path = os.path.join(train_dir, step_name, f'model_{i}')
        self.q_net.save(save_path)
        pd.DataFrame(num_actions).to_csv(os.path.join(train_dir, step_name, f'steps_{i}.csv'))
        return save_path

    def align_target_model(self):
        """
        Sets the target net weights to be the same as the q-net.
        """
        self.target_net.set_weights(self.q_net.get_weights())

    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        Given a state and legal actions that can be taken from that step, calculates the Q-values of (state, action) for
        each action that can be taken (illegal actions are evaluated as 0). Also works for a batch of states.
        :param states: a batch of states
        :return: A numpy.ndarray representing the Estimated Q-values of all actions that can be taken from the specified
                state
        """

        return self.q_net.predict(tf.convert_to_tensor(states))

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        """
        Updates the net according to the Double Q-learning paradigm.
        :param states: A batch of states.
        :param actions: A batch of actions taken from the specified states.
        :param next_states: the observed next-states after taking the specified actions from the specified states
                (Expects None if the state was a terminal state).
        """
        targets = self.q_net.predict(states)
        # Create masks for separating terminal states from non terminal states.
        terminal_mask = np.array([next_state is None for next_state in list(next_states)])
        non_terminal_mask = np.logical_not(terminal_mask)

        # Update the expected sum of rewards in the terminal states to be just the current reward
        if terminal_mask.size > 0:
            targets[terminal_mask, actions[terminal_mask]] = self.punishment

        # Calculate the expected sum of rewards based on the target net and q net
        t = self.target_net.predict(np.asarray(list(next_states[non_terminal_mask])))
        q = self.q_net.predict(np.asarray(list(next_states[non_terminal_mask])))

        # Double-Q learning paradigm. We choose the actions based on the q_net evaluation, but evaluate those chosen
        # actions using the target net
        max_actions = np.argmax(q, axis=-1)
        estimated_values = t[np.arange(t.shape[0]), max_actions]
        targets[non_terminal_mask, actions[non_terminal_mask]] = \
            self.discount * estimated_values + 1 + (actions[non_terminal_mask] == 2).astype(int)

        # At this point the target is similar to the q-net prediction, except in the index corresponding to action
        # taken. In this index, the target value is just the reward if state is terminal, otherwise it is
        # reward + discount * Q(next_state, action), where Q(next_state, action) is evaluated using the Double
        # Q-learning algorithm. that is Q(next_state, action) = t_net(next_state)[argmax(q_net(next_state))]

        self.q_net.fit(states, targets, epochs=5, verbose=0, batch_size=self.batch_size, )

    def set_model(self, model):
        self.q_net = model
        self.target_net = clone_model(self.q_net)
        self.target_net.set_weights(self.q_net.get_weights())
