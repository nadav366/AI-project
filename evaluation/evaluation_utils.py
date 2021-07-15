import os
import json
import sys

import numpy as np
import pandas as pd
import multiprocessing

from game.training_environment import TrainingEnv


def one_fight(players: list, num_of_fights=1):
    num_of_win = dict.fromkeys(range(len(players)), 0)
    for _ in range(num_of_fights):
        env = TrainingEnv(players, training_mode=True, arena_size=400)
        winner_idx = env.loop()
        if winner_idx != -1:
            num_of_win[winner_idx] += 1

    return {k: v / num_of_fights for k, v in num_of_win.items()}


def fights(i, save_path, step_name, train_dir, step_index):
    players_dict = {'single_random': ['r'],
                    'two_random': ['r', 'r'],
                    'single_old': ['old'],
                    'two_old': ['old', 'old'],
                    'old_random': ['old', 'r']}
    result_dict = {
        'name': step_name,
        'i': i,
        'step_index': step_index,
    }
    for run, players in players_dict.items():
        game_result = one_fight([save_path] + players)
        result_dict[run] = game_result[0]

    rand_csv_path = os.path.join(train_dir, 'results.csv')
    df = read_or_create(rand_csv_path)
    df = df.append(result_dict, ignore_index=True)
    df.to_csv(rand_csv_path, index=False)


def fight_checkpoints(run_path):
    with open(os.path.join(run_path, 'params.json')) as json_file:
        json_dict = json.load(json_file)
    plan = json_dict['train_plan']
    f = lambda s: eval(s.split('.')[0].split('_')[-1])
    for step_index, step in enumerate(plan):
        # step = {players, num of games, arena size, first not random, name}
        specific_run_path = os.path.join(run_path, step['des'])
        for model_name in sorted(os.listdir(specific_run_path), key=f):
            model_path = os.path.join(specific_run_path, model_name)
            if not os.path.isdir(model_path):
                continue
            fights(i=f(model_name),
                   save_path=model_path,
                   step_name=step['des'],
                   train_dir=run_path,
                   step_index=step_index)


# def fights(i, save_path, step_name, train_dir, step_index):
#     try:
#         res_rand = one_fight([save_path, 'r'])
#         rand_csv_path = os.path.join(train_dir, 'random.csv')
#         df = read_or_create(rand_csv_path)
#         df = df.append({
#             'name': step_name,
#             'i': i,
#             'train_model': res_rand[0],
#             'ref_model': res_rand[1],
#             'step_index': step_index}, ignore_index=True)
#         df.to_csv(rand_csv_path, index=False)
#     except Exception as e:
#         print('Exception in random fight')
#         print(e)
#
#     try:
#         old_rand = one_fight([save_path, 'old'])
#         rand_csv_path = os.path.join(train_dir, 'old.csv')
#         df = read_or_create(rand_csv_path)
#         df = df.append({
#             'name': step_name,
#             'i': i,
#             'train_model': old_rand[0],
#             'ref_model': old_rand[1],
#             'step_index': step_index}, ignore_index=True)
#         df.to_csv(rand_csv_path, index=False)
#     except Exception as e:
#         print('Exception in old fight')
#         print(e)

def create_results_csv(runs_path):
    relevant_runs = []
    for run in os.listdir(runs_path):
        abs_path = os.path.join(runs_path, run)
        if os.path.exists(os.path.join(abs_path, 'final_model')) and not os.path.exists(
                os.path.join(abs_path, 'results.csv')):
            relevant_runs.append(abs_path)

    with multiprocessing.Pool(3) as pool:
        pool.imap(fight_checkpoints, relevant_runs)


def read_or_create(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    return df


if __name__ == '__main__':
    create_results_csv(sys.argv[1])
