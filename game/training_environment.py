### IMPORTS ###
import numpy as np
from game.players.player_factory import PlayerFactory
from game.achtung_environment import AchtungEnv
from game.players.drl_player import DRLPlayer


class TrainingEnv(AchtungEnv):
    def __init__(self, players, training_mode=False, with_positions=True):
        AchtungEnv.__init__(self, training_mode)
        self.initialize(players)
        self.use_positions = with_positions

    def get_state(self, player_id=0):
        return self.state.adjust_to_drl_player(player_id)

    def set_player(self, player_id, *args):
        self.players[player_id] = DRLPlayer(player_id, self, *args)

    def get_legal_actions(self, state=None, player_id=0):
        return np.ones(3)

    def step(self, action, player_id=0):
        if not self.state.alive[player_id]:  # or self.state.is_terminal_state():
            return None, 0
        for i, player in enumerate(self.players):
            if self.state.alive[i]:
                if i == player_id:
                    self.actions[i] = action
                else:
                    self.actions[i] = player.get_action(self.state)
        for _ in range(self.action_sampling_rate):
            self.tick()
        return self.get_state(player_id), 1

    def loop(self):
        while True:
            self.counter += 1
            self.update_actions()
            self.tick()
            if np.sum(self.state.alive) <= 1:
                winner_idx = np.where(self.state.alive)[0]
                if len(winner_idx) > 0:
                    return winner_idx[0]
                else:
                    return -1

