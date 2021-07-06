import datetime
import json
import os
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, Flatten, BatchNormalization

from double_dqn.agent import DQNAgent
from game.training_environment import TrainingEnv


def get_model(params: SimpleNamespace):
    n_actions = 3
    input_tensor = Input(shape=(params.state_size, params.state_size, 1))
    x = input_tensor
    for filter_size in params.conv_filters:
        x = Conv2D(filters=filter_size, kernel_size=(3, 3), activation='relu')(x)
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
    return model

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)
    params = SimpleNamespace(**params)
    time = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    dir_train = os.path.join('run_trains', f"{params.name}_{time}")
    os.makedirs(dir_train)

    model = get_model(params)
    exploration_rate = None
    df = pd.DataFrame()
    for step_params in params.train_plan:
        step_params = SimpleNamespace(**step_params)
        print(f'start {step_params.des}')
        players = ['r'] + step_params.players
        game = TrainingEnv(players, training_mode=True, arena_size=step_params.arena_size)
        trained_agent = DQNAgent(game, model, exploration_decay=params.exploration_decay)
        num_actions, exploration_rate = trained_agent.train(step_params.num_of_games,
                                                            dir_train,
                                                            step_name=step_params.des,
                                                            exploration_rate=exploration_rate,
                                                            state_size=params.state_size)
        df = df.append(pd.DataFrame({
            'name': step_params.des,
            'num_actions': num_actions,
            'step': np.arange(len(num_actions))}), ignore_index=True)
        df.to_csv(os.path.join(dir_train, 'df_all.csv'))
        model.save(os.path.join(dir_train, f"{step_params.des}_model"))

    model.save(os.path.join(dir_train, 'final_model'))
