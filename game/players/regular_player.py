from static.settings import *
from game.players.player import Player


class RegularHumanPlayer(Player):

    def __init__(self, player_id, game, right, left):
        Player.__init__(self, player_id, game)
        self.right = right
        self.left = left
        self.temp = 0

    def get_action(self, state):
        keys = pygame.key.get_pressed()
        if keys[self.right]:
            return RIGHT
        if keys[self.left]:
            return LEFT
        return STRAIGHT
