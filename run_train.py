import datetime
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, Flatten, BatchNormalization, MaxPool2D

from double_dqn.agent import DQNAgent
from game.training_environment import TrainingEnv

n_actions = 3


def get_model(params: SimpleNamespace):
    if params.model_type == 'fc':
        return get_fc_model(params)
    if params.model_type == 'conv':
        return get_conv_model(params)
    raise Exception(f'Unknown model_type- {params.model_type}')


def get_fc_model(params: SimpleNamespace):
    input_tensor = Input(shape=27)
    x = input_tensor

    for fc_size in params.fc_sizes:
        x = Dense(fc_size, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params.dropout)(x)

    final = Dense(n_actions)(x)
    model = Model(inputs=input_tensor, outputs=final)
    model.compile(optimizer='adam', loss='mse')
    return model, True


def get_conv_model(params: SimpleNamespace):
    input_tensor = Input(shape=(params.state_size, params.state_size, 1))
    x = input_tensor
    for filter_size in params.conv_filters:
        x = Conv2D(filters=filter_size, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(params.dropout)(x)
    x = Flatten()(x)

    for fc_size in params.fc_sizes:
        x = Dense(fc_size, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params.dropout)(x)

    final = Dense(n_actions)(x)
    model = Model(inputs=input_tensor, outputs=final)
    model.compile(optimizer='adam', loss='mse')
    return model, False


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)
    params = SimpleNamespace(**params)
    time = datetime.datetime.now().strftime("%d%m%y_%H%M%S")

    curr_dir_train = getattr(params, 'curr_dir_train', None)
    if curr_dir_train is None:
        dir_train = os.path.join('run_trains', f"{params.model_type}_{params.name}_{time}")
        os.makedirs(dir_train)
        model, extract_features = get_model(params)
        exploration_rate = 1
        df = pd.DataFrame()
    else:
        dir_train = f'{curr_dir_train}_{time}'
        os.makedirs(dir_train)
        model = tf.keras.models.load_model(os.path.join(curr_dir_train, 'final_model'))
        extract_features = len(model.input.shape.as_list()) == 2
        with open(os.path.join(curr_dir_train, 'exploration_rate.txt'), 'r') as f:
            exploration_rate = float(f.readlines()[-1])
        df = pd.read_csv(os.path.join(curr_dir_train, 'df_all.csv'))

    for step_params in params.train_plan:
        step_params = SimpleNamespace(**step_params)
        print(f'start {step_params.des}')
        players = ['r'] + step_params.players
        game = TrainingEnv(players, training_mode=True, arena_size=step_params.arena_size, extract_features=extract_features)
        trained_agent = DQNAgent(game, model, exploration_decay=params.exploration_decay, discount=params.discount)
        num_actions, exploration_rate = trained_agent.train(step_params.num_of_games,
                                                            dir_train,
                                                            step_name=step_params.des,
                                                            exploration_rate=(exploration_rate + 1.0) / 2.0,
                                                            state_size=params.state_size)

        step_index = df['step_index'].max() + 1 if len(df) > 0 else 0
        df = df.append(pd.DataFrame({
            'name': step_params.des,
            'num_actions': num_actions,
            'step': np.arange(len(num_actions)),
            'step_index': step_index}), ignore_index=True)
        df.to_csv(os.path.join(dir_train, 'df_all.csv'))
        model.save(os.path.join(dir_train, f"{step_params.des}_model"))

    model.save(os.path.join(dir_train, 'final_model'))
