import os

import numpy as np
import pandas as pd

from game.training_environment import TrainingEnv


def one_fight(players: list, num_of_fights=50):
    num_of_win = dict.fromkeys(range(len(players)), 0)
    for _ in range(num_of_fights):
        env = TrainingEnv(players, training_mode=True, arena_size=400)
        winner_idx = env.loop()
        if winner_idx != -1:
            num_of_win[winner_idx] += 1
    num_of_wins_fights = np.sum(list(num_of_win.values()))
    return {k: v / num_of_wins_fights for k, v in num_of_win.items()}


def fights(i, save_path, step_name, train_dir, step_index):
    try:
        res_rand = one_fight([save_path, 'r'])
        rand_csv_path = os.path.join(train_dir, 'random.csv')
        df = read_or_create(rand_csv_path)
        df = df.append({
            'name': step_name,
            'i': i,
            'train_model': res_rand[0],
            'ref_model': res_rand[1],
            'step_index': step_index}, ignore_index=True)
        df.to_csv(rand_csv_path, index=False)
    except Exception as e:
        print('Exception in random fight')
        print(e)

    try:
        old_rand = one_fight([save_path, 'old'])
        rand_csv_path = os.path.join(train_dir, 'old.csv')
        df = read_or_create(rand_csv_path)
        df = df.append({
            'name': step_name,
            'i': i,
            'train_model': old_rand[0],
            'ref_model': old_rand[1],
            'step_index': step_index}, ignore_index=True)
        df.to_csv(rand_csv_path, index=False)
    except Exception as e:
        print('Exception in old fight')
        print(e)


def read_or_create(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    return df