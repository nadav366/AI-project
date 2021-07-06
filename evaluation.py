from game.training_environment import TrainingEnv
import numpy as np

def fight(players: list, num_of_fights=50):
    num_of_win = dict.fromkeys(range(len(players)), 0)
    for _ in range(num_of_fights):
        env = TrainingEnv(players, training_mode=True)
        winner_idx = env.loop()
        if winner_idx != -1:
            num_of_win[winner_idx] += 1
    num_of_wins_fights = np.sum(list(num_of_win.values()))
    return {k: v / num_of_wins_fights for k, v in num_of_win.items()}
