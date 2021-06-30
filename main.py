from game.achtung_environment import AchtungEnv
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_outcomes(rewards, path, width, height, random_initialization, positions):
    plot_average_rewards_over_100(rewards, path, width, height, random_initialization, positions)


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=int)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_average_rewards_over_100(rewards, path, width, height, random_initialization, positions):
    r = moving_average(rewards)
    plt.plot(r[::50], label=f'Moving average')
    plt.legend()
    plt.ylabel('Average over last 100 episodes')
    plt.xlabel('Episodes')
    plt.xticks(np.arange(0, len(r) // 50, 10), [i * 500 for i in range(len(r // 50))])
    title = 'Moving average rewards of past 100 training episodes \n' + f'Arena shape - ({width},{height})'
    if random_initialization:
        title += ' with random initialization'
    else:
        title += ' with constant initialization'
    plt.title(title)
    plt.savefig(path + os.path.sep + 'moveing_average')
    plt.show()


def plot_rewards(rewards, path, width, height, random_initialization, positions):
    pass


if __name__ == '__main__':
    # saving_path = r"C:\Users\danie\Studies\B.Sc\year3\Semester B\67842 - Introduction to Artificial Intelligence\Project\AchtungDeKurve\static\models\fc_model\small_net_with_position\model_1\large_arena_initial_try"
    # my_player = 52.3324
    # random_player = 100 - my_player
    # plt.bar([1, 2], [my_player, random_player], width=0.5, color=['orange', 'blue'])
    # plt.ylim(0, 80)
    # plt.title('DRL player comparison')
    # plt.ylabel('Percentage of wins')
    # plt.xticks([1, 2], ['DRL player', 'random player'])
    # plt.xlim(0.5, 2.5)
    # plt.show()
    # rewards = []
    # for i in range(1, 3):
    #     path = saving_path + os.path.sep + f'session_{i}_small_arena' + os.path.sep + 'checkpoints' + os.path.sep + 'total_rewards.csv'
    #     with open(path) as file:
    #         rewards.append(pd.read_csv(file))
    # rewards = pd.concat(rewards)
    # with open(saving_path + os.path.sep + 'checkpoints' + os.path.sep + 'total_rewards.csv', 'r') as file:
    #     rewards = pd.read_csv(file)
    # plot_outcomes(np.array(rewards[:]['0']), saving_path, 350, 350, random_initialization=True, positions = True)
    game = AchtungEnv()
    game.play()
