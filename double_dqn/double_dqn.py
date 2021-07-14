# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import clone_model, load_model, model_from_json
#
#
# class DoubleDQN:
#     def __init__(self, final_conv_model, discount: float = 0.95):
#         self.discount = discount
#         self.q_net = None
#         self.target_net = None
#         self.set_model(final_conv_model)
#
#     def align_target_model(self):
#         """
#         Sets the target net weights to be the same as the q-net.
#         """
#         self.target_net.set_weights(self.q_net.get_weights())
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
#         return self.q_net.predict(tf.convert_to_tensor(states))
#
#     def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
#         """
#         Updates the net according to the Double Q-learning paradigm.
#         :param states: A batch of states.
#         :param actions: A batch of actions taken from the specified states.
#         :param next_states: the observed next-states after taking the specified actions from the specified states
#                 (Expects None if the state was a terminal state).
#         """
#         targets = self.q_net.predict(states)
#         # Create masks for separating terminal states from non terminal states.
#         terminal_mask = np.array([next_state is None for next_state in list(next_states)])
#         non_terminal_mask = np.logical_not(terminal_mask)
#
#         # Update the expected sum of rewards in the terminal states to be just the current reward
#         if terminal_mask.size > 0:
#             targets[terminal_mask, actions[terminal_mask]] = \
#                 targets[terminal_mask, actions[terminal_mask]] * self.discount - 10
#
#         # Calculate the expected sum of rewards based on the target net and q net
#         t = self.target_net.predict(np.asarray(list(next_states[non_terminal_mask])))
#         q = self.q_net.predict(np.asarray(list(next_states[non_terminal_mask])))
#
#         # Double-Q learning paradigm. We choose the actions based on the q_net evaluation, but evaluate those chosen
#         # actions using the target net
#         max_actions = np.argmax(q, axis=-1)
#         estimated_values = t[np.arange(t.shape[0]), max_actions]
#         targets[non_terminal_mask, actions[non_terminal_mask]] = self.discount * estimated_values + 1
#
#         # At this point the target is similar to the q-net prediction, except in the index corresponding to action
#         # taken. In this index, the target value is just the reward if state is terminal, otherwise it is
#         # reward + discount * Q(next_state, action), where Q(next_state, action) is evaluated using the Double
#         # Q-learning algorithm. that is Q(next_state, action) = t_net(next_state)[argmax(q_net(next_state))]
#
#         self.q_net.fit(states, targets, epochs=10, verbose=0)
#
#     def set_model(self, final_conv_model):
#         self.q_net = final_conv_model
#         self.target_net = clone_model(self.q_net)
#         self.target_net.set_weights(self.q_net.get_weights())
