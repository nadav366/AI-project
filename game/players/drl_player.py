import numpy as np

from game.players.player import Player


class DRLPlayer(Player):
    def __init__(self, player_id, game, model, extract_features=False):
        super().__init__(player_id, game)

        self._net = model
        self.predictions = 0
        self.total_time = 0
        self.extract_features=extract_features

    def get_action(self, state):
        self.predictions += 1
        drl_state = state.adjust_to_drl_player(self.id)  # self.crop_box(state.board, state.positions)
        values = self._net(drl_state[np.newaxis, ...], training=False)
        return np.random.choice(np.flatnonzero(values == np.max(values)))
