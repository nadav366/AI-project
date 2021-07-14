import tensorflow as tf

from game.players.drl_player import DRLPlayer
from game.players.random_player import RandomPlayer
from game.players.regular_player import RegularHumanPlayer
from static.settings import *

ARROWS_RIGHT = pygame.K_RIGHT
ARROWS_LEFT = pygame.K_LEFT
WASD_RIGHT = pygame.K_d
WASD_LEFT = pygame.K_a
MIN_MAX_DEPTH = 3


class PlayerFactory:
    @staticmethod
    def create_player(player_type, id, game, extract_features):
        if player_type == 'ha':
            return RegularHumanPlayer(id, game, ARROWS_RIGHT, ARROWS_LEFT)
        elif player_type == 'r':
            return RandomPlayer(id, game, extract_features=extract_features)
        elif player_type == 'conv':
            player_type = os.path.join('models', 'final_conv_model')
        elif player_type == 'old':
            player_type = os.path.join('models', 'old_model_save')
        elif player_type == 'd':
            player_type = os.path.join('models', 'curr_fc_model')
        model = tf.keras.models.load_model(player_type)
        extract_features = len(model.input.shape.as_list()) == 2
        return DRLPlayer(id, game, model, extract_features=extract_features)
