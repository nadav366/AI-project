from game.training_environment import TrainingEnv


def fight(players: list, num_of_fights=100):
    num_of_win = dict.fromkeys(range(len(players)), 0)
    for i in range(num_of_fights):
        env = TrainingEnv(players, training_mode=True)
        winner_idx = env.loop()
        num_of_win[winner_idx] += 1
    return {k: v / num_of_fights for k, v in num_of_win.items()}
