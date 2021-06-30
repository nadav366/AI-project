import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, Flatten

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
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)
    time = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    dir_train = os.path.join('.', f"{params['name']}_{time}")
    os.makedirs(dir_train)

    pd.DataFrame().to_csv(os.path.join(dir_train, 'random.csv'))
    pd.DataFrame().to_csv(os.path.join(dir_train, 'old.csv'))

    model = get_model()
    exploration_rate = None
    df = pd.DataFrame()
    for step_params in params['train_plan']:
        print(f'start {step_params["des"]}')
        players = ['r'] + step_params['players']
        game = TrainingEnv(players, training_mode=True)
        trained_agent = DQNAgent(game, model)
        num_actions, exploration_rate = trained_agent.train(step_params['num_of_games'], dir_train,
                                                            step_name=step_params['des'],
                                                            exploration_rate=exploration_rate)
        df = df.append(pd.DataFrame({
            'name': step_params['des'],
            'num_actions': num_actions,
            'step': np.arange(len(num_actions))}), ignore_index=True)
        df.to_csv(os.path.join(dir_train, 'df_all.csv'))
        model.save(os.path.join(dir_train, f"{step_params['des']}_model"))

    model.save(os.path.join(dir_train, 'final_model'))
