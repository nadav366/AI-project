import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_graph_for_all_train(dir_path):
    all_df_path = os.path.join(dir_path, 'df_all.csv')
    if os.path.exists(all_df_path):
        df = pd.read_csv(all_df_path)
    else:
        try:
            one_stage_name = sorted([f for f in os.listdir(dir_path) if 'steps' in f])[-1]
        except:
            print(dir_path)
            # shutil.rmtree(dir_path)
            return
        one_stage_path = os.path.join(dir_path, one_stage_name)
        df = pd.read_csv(one_stage_path)
        df['step_index'] = 0
        df['name'] = os.path.basename(dir_path)
        df.rename({'Unnamed: 0': 'step', '0': 'num_actions'}, axis=1, inplace=True)

    plt.figure(figsize=(max(4, 4 * len(df) / 1000), 4))
    df.reset_index(drop=True, inplace=True)
    groups = df.groupby('step_index')
    num_actions_arr = groups.num_actions.apply(list)
    name_arr = groups.name.unique()
    steps_arr = groups.step.apply(list)
    N = 100
    last_step_len = 0
    y_all = np.convolve(df.num_actions, np.ones((N,)) / N, mode='valid')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for i in range(0, steps_arr.size):
        y_smooth = np.convolve(num_actions_arr[i], np.ones((N,)) / N, mode='valid')
        x_vals = last_step_len + (np.arange(len(y_smooth)) + N // 2)
        plt.plot(x_vals, y_smooth, label=name_arr[i][0])
        y_pos = get_y_text_pos(y_all, y_smooth)
        plt.text((last_step_len + x_vals.mean()) // 2, y_pos, name_arr[i][0], bbox=props)
        last_step_len += len(num_actions_arr[i])
        if i + 1 != steps_arr.size:
            plt.axvline(x=last_step_len, ls='dotted')

    plt.xlabel('steps')
    plt.ylabel('actions')
    plt.title(f'Model Evaluation, moving average on {N} games\n For: {os.path.basename(dir_path)}')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    plt.savefig(os.path.join(dir_path, 'actions.png'))
    plt.close()


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


def plot_graph_for_comparing(compare_path, dir_path, name):
    df = pd.read_csv(compare_path)
    df.reset_index(drop=True, inplace=True)
    cp_rate = df['i'].min() + 1
    df['name'] = df['name'].fillna('')
    groups = df.groupby('step_index')
    iter_num_arr = groups.i.apply(list)
    name_arr = groups.name.unique().apply(lambda arr: arr[0])

    player_1_wins_arr = df['train_model']
    player_2_wins_arr = df['ref_model']
    group_last_iter = 0

    plt.plot((np.arange(len(df)) + 1) * cp_rate, player_1_wins_arr, label="train_model")
    # plt.plot((np.arange(len(df)) + 1) * cp_rate, player_2_wins_arr, label=f'{name}_model', ls='dashed')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(iter_num_arr.size):
        if name_arr[i] != '':
            plt.text((group_last_iter + iter_num_arr[i][-1] // 3), 0.5, name_arr[i], bbox=props)
        group_last_iter += iter_num_arr[i][-1]
        if i + 1 < iter_num_arr.size:
            plt.axvline(x=group_last_iter, ls='dotted')
    plt.ylim(0, 1)
    plt.xlabel('num of iter')
    plt.ylabel('win %')

    plt.title(f'Compare {name} to train\n{os.path.basename(dir_path)}')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(dir_path, f'{name}.png'))
    plt.close()


def plot_compared_games(compare_path, dir_path, name):
    df = pd.read_csv(compare_path)
    df.reset_index(drop=True, inplace=True)
    cp_rate = df['i'].min() + 1
    df['name'] = df['name'].fillna('')
    groups = df.groupby('step_index')
    iter_num_arr = groups.i.apply(list)
    name_arr = groups.name.unique().apply(lambda arr: arr[0])

    for column in df.columns:
        if column not in ['i', 'step_index', 'name']:
            wins_arr = df[column]
            group_last_iter = 0

            plt.plot((np.arange(len(df)) + 1) * cp_rate, wins_arr, label=column)

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            for i in range(iter_num_arr.size):
                if name_arr[i] != '':
                    plt.text((group_last_iter + iter_num_arr[i][-1] // 3), 0.5, name_arr[i], bbox=props)
                group_last_iter += iter_num_arr[i][-1]
                if i + 1 < iter_num_arr.size:
                    plt.axvline(x=group_last_iter, ls='dotted')
    plt.ylim(0, 1)
    plt.xlabel('num of iter')
    plt.ylabel('win %')

    plt.title(f'Compare {name} to train\n{os.path.basename(dir_path)}')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(dir_path, f'{name}.png'))
    plt.close()


def main(root_dir):
    for run_dir in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_dir)

        plot_graph_for_all_train(run_path)

        results_path = os.path.join(run_path, 'results.csv')
        if os.path.exists(results_path):
            plot_compared_games(results_path, dir_path=run_path, name='result')


if __name__ == '__main__':
    main(sys.argv[1])
