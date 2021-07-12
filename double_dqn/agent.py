import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model, model_from_json
from tqdm import tqdm

from double_dqn.double_dqn import DoubleDQN
from double_dqn.experience_replay import ExperienceReplay
from game.training_environment import TrainingEnv
from evaluation import fight

class DQNAgent:
    def __init__(self, env, model: tf.keras.models.Model,
                 net_update_rate: int = 10,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 1e-6):
        # set hyper parameters
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.net_updating_rate = net_update_rate

        # set environment
        self.env = env
        self.state_shape = tuple(model.input.shape.as_list()[1:])
        self.action_shape = tuple(model.output.shape.as_list()[1:])

        # the number of experience per batch for batch learning
        # Experience Replay for batch learning
        self.exp_rep = ExperienceReplay(state_size=self.state_shape[0])

        # Deep Q Network
        self.net = DoubleDQN(model)

    def get_action(self, state: np.ndarray, eps) -> int:
        """Given a state returns a random action with probability eps, and argmax(q_net(state)) with probability 1-eps.
           (only legal actions are considered)"""

        if np.random.random() >= eps:  # Exploitation
            values = self.net.predict(state[np.newaxis, ...])
            # Calculate the Q-value of each action
            # probs = tf.math.softmax(q).numpy().flatten()
            # choice = np.random.choice(a=[0, 1, 2], p=probs)
            choice = np.random.choice(np.flatnonzero(values == np.max(values)))
            return choice
        return np.random.choice(np.arange(3))

    def update_net(self, batch_size: int):
        """ if there are more than batch_size experiences, Optimizes the network's weights using the Double-Q-learning
         algorithm with a batch of experiences, else returns"""
        if self.exp_rep.get_num() < batch_size:
            return
        batch = self.exp_rep.get_batch(batch_size)
        self.net.fit(*batch)

    def train(self, episodes: int, train_dir, step_name,
              max_actions: int = None, batch_size: int = 64,
              checkpoint_rate=300, exploration_rate=None):
        """
        Runs a training session for the agent
        :param episodes: number of episodes to train.
        :param max_actions: max number of steps in an episode. if 0, each episode runs until reaching a terminal state.
        :param batch_size: number of experiences to learn from in each net_update.
        :return A tuple containing 2 lists. the first is a list of accumulated rewards for each training episode.
                The second list contains the number of actions taken in each training episode.
        """
        if self.net is None:
            raise NotImplementedError('agent.train called before model was not initiated. Please set the agent\'s model'
                                      ' using the set_model method. You can access the state and action shapes using '
                                      'agent\'s methods \'get_state_shape\' and \'get_action_shape\'')

        # set hyper parameters
        exploration_rate = self.exploration_rate if exploration_rate is None else exploration_rate
        total_rewards = []
        num_actions = []
        # start training
        for i in tqdm(range(episodes)):
            self.env.reset()  # Reset the environment for a new episode
            state = self.env.get_state()  # Get starting state
            step = 0
            ep_reward = 0
            while max_actions is None or step <= max_actions:
                step += 1
                action = self.get_action(state, exploration_rate)
                next_state, reward = self.env.step(action)
                # Add experience to memory
                self.exp_rep.add(state, action, next_state)
                self.update_net(batch_size)  # Optimize the DoubleQ-net
                if next_state is None:  # The action taken led to a  terminal state
                    break
                if (step % self.net_updating_rate) == 0:
                    # update target network
                    self.net.align_target_model()

                state = next_state

            # Update total_rewards and num_actions to keep track of progress
            num_actions.append(step)
            # Update target network at the end of the episode
            self.net.align_target_model()  # Optimize the DoubleQ-net
            if (step % self.net_updating_rate) == 0:
                # update target network
                self.net.align_target_model()

            if self.exp_rep.get_num() > batch_size:
                if (i % checkpoint_rate) == checkpoint_rate - 1:
                    save_path = self.save_data_of_cp(num_actions, step_name, train_dir, i)
                    # self.fights(i, save_path, step_name, train_dir, step_index)
            with open(os.path.join(train_dir, 'exploration_rate.txt'), 'a') as exp_rate:
                exp_rate.writelines([str(exploration_rate) + '\n'])

            # Update exploration rate
            exploration_rate = max(0.1, 0.01 + (exploration_rate - 0.01) * np.exp(-self.exploration_decay * (i + 1)))
        self.exploration_rate = exploration_rate
        return num_actions, exploration_rate

    def save_data_of_cp(self, num_actions, step_name, train_dir, i):
        save_path = os.path.join(train_dir, step_name, f'model_{i}')
        self.save_model(save_path)
        pd.DataFrame(num_actions).to_csv(os.path.join(train_dir, step_name, f'steps_{i}.csv'))
        return save_path

    def fights(self, i, save_path, step_name, train_dir, step_index):
        try:
            res_rand = fight([save_path, 'r'])
            rand_csv_path = os.path.join(train_dir, 'random.csv')
            df = self.read_or_create(rand_csv_path)
            df = df.append({'name': step_name, 'i': i, 'me': res_rand[0], 'rand_res': res_rand[1], 'step_index': step_index}, ignore_index=True)
            df.to_csv(rand_csv_path, index=False)
        except Exception as e:
            print('Exception in random fight')
            print(e)

        try:
            old_rand = fight([save_path, 'old'])
            rand_csv_path = os.path.join(train_dir, 'old.csv')
            df = self.read_or_create(rand_csv_path)
            df = df.append({'name': step_name, 'i': i, 'me': old_rand[0], 'old_player': old_rand[1], 'step_index': step_index}, ignore_index=True)
            df.to_csv(rand_csv_path, index=False)
        except Exception as e:
            print('Exception in old fight')
            print(e)


    @staticmethod
    def read_or_create(path):
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()
        return df

    def get_state_shape(self):
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape

    def set_model(self, model):
        """ """
        self.net = DoubleDQN(model)

    # Handles saving\loading the model as explained here: https://www.tensorflow.org/guide/keras/save_and_serialize
    def load_weights(self, path):
        self.net.load_weights(path)

    def save_weights(self, path):
        self.net.save_weights(path)

    def save_model(self, path):
        if self.net is None:
            raise NotImplementedError('agent.save_model was called before model was not initiated. Please set the '
                                      'agent\'s model using the set_model method. You can access the state and action '
                                      'shapes using agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        self.net.save_model(path)

    def load_model(self, path):
        model = load_model(path)
        self.set_model(model)

    def to_json(self, **kwargs):
        if self.net is None:
            raise NotImplementedError('agent.to_json was called before model was not initiated. Please set the '
                                      'agent\'s model using the set_model method. You can access the state and action '
                                      'shapes using agent\'s methods \'get_state_shape\' and \'get_action_shape\'')
        return self.net.to_json(**kwargs)

    def from_json(self, json_config):
        model = model_from_json(json_config)
        self.set_model(model)



