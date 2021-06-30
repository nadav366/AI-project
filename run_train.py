import pandas as pd
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from matplotlib import pyplot as plt

from double_dqn.agent import DQNAgent
from game.training_environment import TrainingEnv

train_plan = [
    ([], 100, 'name')
]
weight_path = '.tmp'
training_data_path = '.tmp'


def get_model():
    n_actions = 3
    input_tensor = Input(shape=(32, 32, 1))
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_tensor)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    final = Dense(n_actions)(x)
    model = Model(inputs=input_tensor, outputs=final)
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    # run_name =
    model = get_model()
    for players, num_of_eps, name in train_plan:
        players = ['r'] + players
        game = TrainingEnv(players, training_mode=True)
        trained_agent = DQNAgent(game, model)
        rewards, num_actions = trained_agent.train(num_of_eps, weight_path, training_data_path, None, 64)
        pd.DataFrame(rewards).to_csv('rewards.csv')
        model.save('model')
