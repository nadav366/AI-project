import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fontsize = 17


def name2step(name):
    return int(name.replace('steps_', '').replace('.csv', ''))


def plot_graph_for_all_train(dir_path, last_name, ax):
    all_df_path = os.path.join(dir_path, 'df_all.csv')
    if os.path.exists(all_df_path):
        df = pd.read_csv(all_df_path)
        if last_name is not None and last_name != '' and last_name not in df['name']:
            step_path = os.path.join(dir_path, last_name)
            if os.path.exists(step_path):
                # print(step_path)
                df_name = sorted([f for f in os.listdir(step_path) if 'steps' in f], key=name2step)[-1]
                df_path = os.path.join(step_path, df_name)
                df_to_add = pd.read_csv(df_path)
                df_to_add['step_index'] = df['step_index'].max() + 1
                df_to_add['name'] = last_name
                df_to_add.rename({'Unnamed: 0': 'step', '0': 'num_actions'}, axis=1, inplace=True)
                df = df.append(df_to_add, ignore_index=True)
    else:
        try:
            one_stage_name = sorted([f for f in os.listdir(dir_path) if 'steps' in f], key=name2step)[-1]
        except:
            print(dir_path)
            # shutil.rmtree(dir_path)
            return
        one_stage_path = os.path.join(dir_path, one_stage_name)
        df = pd.read_csv(one_stage_path)
        df['step_index'] = 0
        df['name'] = os.path.basename(dir_path)
        df.rename({'Unnamed: 0': 'step', '0': 'num_actions'}, axis=1, inplace=True)

    df['name'] = df['name'].fillna('')
    df.reset_index(drop=True, inplace=True)
    groups = df.groupby('step_index')
    num_actions_arr = groups.num_actions.apply(list)
    name_arr = groups.name.unique().apply(lambda arr: arr[0])
    N = 100
    last_step_len = 0
    y_all = np.convolve(df.num_actions, np.ones((N,)) / N, mode='valid')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in name_arr.index:
        y_smooth = np.convolve(num_actions_arr[i], np.ones((N,)) / N, mode='valid')
        x_vals = last_step_len + (np.arange(len(y_smooth)) + N // 2)
        plt.plot(x_vals, y_smooth, label=name_arr[i])
        if name_arr[i] != '':
            y_pos = get_y_text_pos(y_all, y_smooth)
            plt.text((last_step_len + x_vals.mean()) // 2, y_pos, name_arr[i], bbox=props)
        last_step_len += len(num_actions_arr[i])
        if i < name_arr.index.max():
            plt.axvline(x=last_step_len, ls='dotted')

    ax.set_xlabel('Games', fontsize=fontsize)
    ax.set_ylabel('Score (Number of steps)', fontsize=fontsize)
    ax.set_title(f'Model Evaluation, moving average on {N} games', fontsize=fontsize)


def get_y_text_pos(y_all, y_smooth):
    min_diss = abs(y_all.min() - y_smooth.min())
    max_diss = abs(y_all.max() - y_smooth.max())
    if max_diss > min_diss:
        y_pos = np.random.randint(y_all.max() * 0.8, y_all.max())
    elif max_diss < min_diss:
        y_pos = np.random.randint(y_all.min(), y_all.min() * 1.3)
    else:
        y_pos = np.random.randint(y_all.min(), y_all.max())
    return y_pos


def plot_compared_games(compare_path, ax):
    df = pd.read_csv(compare_path)
    keep_inx = df[['i', 'name']].drop_duplicates(keep='last').index
    df = df.loc[keep_inx]

    df.reset_index(drop=True, inplace=True)
    cp_rate = df['i'].min() + 1
    df['name'] = df['name'].fillna('')
    groups = df.groupby('step_index')
    iter_num_arr = groups.i.apply(list)
    name_arr = groups.name.unique().apply(lambda arr: arr[0])
    N = 2

    for column in df.columns:
        if column not in ['i', 'step_index', 'name']:
            wins_arr = df[column]
            group_last_iter = 0

            curr_wins_arr = np.convolve(wins_arr, np.ones((N,)) / N, mode='valid')
            ax.plot((np.arange(len(curr_wins_arr)) + 2) * cp_rate, curr_wins_arr, label=column)

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            for i in name_arr.index:
                if name_arr[i] != '':
                    ax.text((group_last_iter + iter_num_arr[i][-1] // 3), 0.2, name_arr[i], bbox=props)
                group_last_iter += iter_num_arr[i][-1]
                if i < name_arr.index.max():
                    ax.axvline(x=group_last_iter, ls='dotted')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Games', fontsize=fontsize)
    ax.set_ylabel('Win Rate', fontsize=fontsize)

    ax.set_title(f'Win rate of agent compared to the listed tasks', fontsize=fontsize)
    ax.legend()

    return name_arr.iloc[-1]


def main(root_dir):
    for run_dir in os.listdir(root_dir):
        if 'plots' in run_dir:
            continue
        run_path = os.path.join(root_dir, run_dir)
        # run_path_to_save = os.path.join(root_dir, 'plots', run_dir)
        run_path_to_save = os.path.join(root_dir, run_dir)
        os.makedirs(run_path_to_save, exist_ok=True)
        fog, axs = plt.subplots(1, 2, figsize=(15, 5))

        last_name = None
        results_path = os.path.join(run_path, 'results_v4.csv')
        if os.path.exists(results_path):
            last_name = plot_compared_games(results_path, ax=axs[0])

        plot_graph_for_all_train(run_path, last_name=last_name, ax=axs[1])

        plt.suptitle(f'{os.path.basename(run_path)}', fontsize=fontsize)
        plt.savefig(os.path.join(run_path_to_save, 'all_res.png'))


if __name__ == '__main__':
    main(sys.argv[1])
