import json
import tensorflow as tf
import pygame
from tensorflow.keras.models import model_from_json

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
    def create_player(player_type, id, game):
        if player_type == 'ha':
            return RegularHumanPlayer(id, game, ARROWS_RIGHT, ARROWS_LEFT)
        elif player_type == 'hw':
            return RegularHumanPlayer(id, game, WASD_RIGHT, WASD_LEFT)
        elif player_type == 'old':
            with open(FC_ARCHITECTURE_PATH, 'r') as json_file:
                config = json.load(json_file)
                model = model_from_json(config)
                model.load_weights(FC_WEIGHT_PATH).expect_partial()
            return DRLPlayer(id, game, model, extract_features=True)
        elif player_type == 'r':
            return RandomPlayer(id, game)
        else:
            if player_type == 'd':
                player_type = r"C:\Users\NADAV\.PyCharm2018.1\AI\project\model"
            model = tf.keras.models.load_model(player_type)
            return DRLPlayer(id, game, model)
