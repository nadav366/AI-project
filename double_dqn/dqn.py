# import tensorflow as tf
# from tensorflow.keras.models import clone_model, load_model, model_from_json
# import numpy as np
#
#
# class DQN:
#     def __init__(self, model, discount: float = 0.95):
#         # , optimizer='adam', loss='mse', discount: float = 0.95):
#         """
#         C-tor
#         """
#         self.discount = discount
#         self.q_net = model
#         self.set_model(model)
#
#     def predict(self, states: np.ndarray) -> np.ndarray:
#         """
#         Given a state and legal actions that can be taken from that step, calculates the Q-values of (state, action) for
#         each action that can be taken (illegal actions are evaluated as 0). Also works for a batch of states.
#         :param states: a batch of states
#         :return: A numpy.ndarray representing the Estimated Q-values of all actions that can be taken from the specified
#                 state
#         """
#
#         q_values = self.q_net.predict(tf.convert_to_tensor(states))
#         return q_values
#
#     def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray):
#         """
#         Updates the net according to the Double Q-learning paradigm.
#         :param states: A batch of states.
#         :param actions: A batch of actions taken from the specified states.
#         :param next_states: the observed next-states after taking the specified actions from the specified states
#                 (Expects None if the state was a terminal state).
#         :param rewards: A batch of rewards given after taking soecified actions from specified states and transitioning
#                 to specified next_states.
#         """
#
#         next_state_actions = self.q_net(next_states)
#
#         y = tf.add(rewards, tf.multiply(self.discount, tf.reduce_max(next_state_actions, axis=1)))
#         q = tf.reduce_sum(self.q_net(states) * tf.one_hot(actions, 3), axis=1)
#
#         tf.stop_gradient(y)
#
#         self.q_net.fit(states, targets, epochs=10, verbose=0)
#
#
#     def load_weights(self, path):
#         self.q_net.load_weights(path)
#         self.target_net = clone_model(self.q_net)
#
#     def save_weights(self, path):
#         self.q_net.save_weights(path)
#
#     def to_json(self, **kwargs):
#         return self.q_net.to_json(**kwargs)
#
#     def from_json(self, json_config):
#         self.set_model(model_from_json(json_config))
#
#     def save_model(self, path):
#         self.q_net.save(path)
#
#     def load_model(self, path):
#         self.set_model(load_model(path))
