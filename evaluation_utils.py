import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from game.training_environment import TrainingEnv


def name2step(name):
    return int(name.replace('model_', '').replace('.csv', ''))


def one_fight(players: list, num_of_fights=50, arena_size=400):
    num_of_win = dict.fromkeys(range(len(players)), 0)
    env = TrainingEnv(players, training_mode=True, arena_size=arena_size)
    for _ in range(num_of_fights):
        env.reset()
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

    rand_csv_path = os.path.join(train_dir, 'results_s200.csv')
    df = read_or_create(rand_csv_path)
    df = df.append(result_dict, ignore_index=True)
    df.to_csv(rand_csv_path, index=False)


def fight_checkpoints(run_path):
    print(f'Start {run_path}')
    with open(os.path.join(run_path, 'params.json')) as json_file:
        json_dict = json.load(json_file)
    plan = json_dict['train_plan']
    for step_index, step in enumerate(plan):
        # step = {players, num of games, arena size, first not random, name}
        specific_run_path = os.path.join(run_path, step['des'])
        print(specific_run_path)
        if not os.path.exists(specific_run_path):
            continue
        for model_name in sorted([f for f in os.listdir(specific_run_path) if 'model_' in f], key=name2step):
            model_path = os.path.join(specific_run_path, model_name)
            if not os.path.isdir(model_path):
                continue
            print(f'{os.path.basename(run_path)}: {step["des"]} - {model_name}')
            fights(i=name2step(model_name),
                   save_path=model_path,
                   step_name=step['des'],
                   train_dir=run_path,
                   step_index=step_index)
    print(f'Finish {run_path}')


def create_results_csv(runs_path):
    relevant_runs = []
    for run in os.listdir(runs_path):
        abs_path = os.path.join(runs_path, run)
        if 'plots' not in run:
            relevant_runs.append(abs_path)

    print(relevant_runs)
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(fight_checkpoints, relevant_runs)

    # map(fight_checkpoints, relevant_runs)


def read_or_create(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    return df


if __name__ == '__main__':
    # create_results_csv(sys.argv[1])
    fight_checkpoints(sys.argv[1])
