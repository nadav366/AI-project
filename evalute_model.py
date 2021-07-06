import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIRS_ROOT = os.path.join(BASE_DIR, 'run_dirs')
RANDOM_CSV = os.path.join(RUN_DIRS_ROOT, 'play_alone_030721_222033/random.csv')
ALL_CSV = os.path.join(RUN_DIRS_ROOT, 'play_alone_030721_222033/df_all.csv')
OLD_CSV = os.path.join(RUN_DIRS_ROOT, 'play_alone_030721_222033/old.csv')
# RANDOM_CSV = 'C://Users/guykatz/PycharmProjects/AchtungDeKurve/AI-project/run_dirs/play_alone_030721_222033/random.csv'


# def read_all_file():
#     df = pd.read_csv(ALL_CSV, sep=',')
#     df.reset_index(drop=True, inplace=True)
#     # index, name, num_actions, step, step_index = df.columns.tolist()
#     index, name, num_actions, step = df.columns.tolist()
#     df['step_index'] = 1
#     df2 = df.copy()
#     df2['step_index'] = 2
#     df2[name] = 'with_players'
#     new_df = df.append(df2)
#     df3 = new_df.copy()
#     df3['step_index'] = 3
#     df3[name] = 'with_players_1'
#     new_df1 = new_df.append(df3)
#     plot_graph_for_all_train(new_df1)


def plot_graph_for_all_train():
    df = pd.read_csv(ALL_CSV, sep=',')
    df.reset_index(drop=True, inplace=True)
    groups = df.groupby('step_index')
    num_actions_arr = groups.num_actions.apply(list)
    name_arr = groups.name.unique()
    steps_arr = groups.step.apply(list)
    N = 100
    last_step_len = 0
    names = []
    for i in range(1, steps_arr.size + 1):
        y_smooth = np.convolve(num_actions_arr[i], np.ones((N,)) / N, mode='valid')
        plt.plot((i-1) * last_step_len + (np.arange(len(y_smooth)) + N // 2), y_smooth, label=name_arr[i][0])
        last_step_len = len(num_actions_arr[i])
        plt.axvline(x=last_step_len, ls='dotted')
        names.append(name_arr[i][0])
    plt.xlabel('steps')
    plt.ylabel('actions')
    name = ', '.join(names)
    plt.title(f'Model Evaluation with moving average\n For: {name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    plt.show()


def plot_graph_for_comparing():
    df = pd.read_csv(OLD_CSV, sep=',')
    df.reset_index(drop=True, inplace=True)
    iter_num, layer_name, player_1, player_2, index = df.columns.tolist()

    groups = df.groupby(index)
    iter_num_arr = groups.i.apply(list)
    name_arr = groups.name.unique()

    player_1_wins_arr = df[player_1]
    player_2_wins_arr = df[player_2]
    group_last_iter = 0

    plt.plot((np.arange(df[iter_num].size) + 1) * iter_num_arr[1][0], player_1_wins_arr, label="ours")
    plt.plot((np.arange(df[iter_num].size) + 1) * iter_num_arr[1][0], player_2_wins_arr, label=player_2, ls='dashed')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(1, iter_num_arr.size + 1):
        plt.text((group_last_iter + iter_num_arr[i][-1] // 3), 0.5, name_arr[i][0],  bbox=props)
        group_last_iter += iter_num_arr[i][-1]
        if i != iter_num_arr.size:
            plt.axvline(x=group_last_iter, ls='dotted')

    plt.xlabel('num of iter')
    plt.ylabel('win %')

    plt.title(f'Model Evaluation\n layer: {layer_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

if __name__ == '__main__':
    # main()
    # read_all_file()
    plot_graph_for_comparing()

