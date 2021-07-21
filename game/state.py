import copy

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
from scipy import ndimage

from static.settings import *


class State:
    count = 0

    def __init__(self, shape, positions, angles, colors, players):
        self.margin = 6
        self.extract_features = [player.extract_features for player in players]
        shape_3d = shape[0] + (self.margin * 2), shape[1] + (self.margin * 2), shape[2]
        shape_2d = (shape[0], shape[1])
        self._rgb_board = np.ones(shape_3d) * WHITE
        self._board = np.ones(shape_2d) * WHITE_2D
        self.arena_size = shape[0]
        self.reset_arena()
        self.colors = colors
        self._positions = positions
        self.alive = [True for _ in range(len(positions))]
        self._angles = angles
        self.counts = [0 for _ in angles]

    def get_3d_pixel(self, coord):
        return self._rgb_board[self.margin + int(round(coord[1])), self.margin + int(round(coord[0])), ...]

    def get_2d_pixel(self, coord):
        return self._board[int(round(coord[1])), int(round(coord[0]))]

    def _get_rgb_board(self):
        return self._rgb_board

    def get_board(self):
        return self._board

    def is_2d_pos_available(self, coord):
        pixel = self.get_2d_pixel(coord)
        return pixel == 0

    # def set_board(self, board: np.ndarray):
    #     arena = self.get_rgb_board()
    #     arena[...] = board.copy()

    def get_position(self, player_id):
        pos = self._positions[player_id]
        return pos

    def get_all_positions(self):
        return self._positions

    def get_all_angles(self):
        return self._angles

    def set_position(self, player_id, position):
        self._positions[player_id] = position

    def adjust_pos_to_board_with_margin(self, position):
        return position[0] + self.margin, position[1] + self.margin

    def get_angle(self, player_id):
        return self._angles[player_id]

    def set_angle(self, player_id, angle):
        self._angles[player_id] = angle

    def reset_arena(self):

        self._rgb_board[self.margin:self.margin + self.arena_size, self.margin:self.margin + self.arena_size, ...] = BLACK
        self._board[: self.arena_size, : self.arena_size] = BLACK_2D

    def adjust_to_drl_player(self, player_id, state_size=32, debug=True):
        if self.extract_features[player_id]:
            max_distance = 350
            num_angles = 25
            features_per_player = 2
            num_features = num_angles + features_per_player
            state_rep = np.zeros(num_features)
            state_rep[:num_angles] = self.get_distance_to_obstacles(self._angles[player_id], self._positions[player_id],
                                                                    num_angles, max_distance)
            state_rep[num_angles: num_angles + features_per_player] = self.get_player_drl_features(player_id)
            if debug:
                print(state_rep)

            return state_rep
        else:
            N = state_size * 2
            crop_size_w = state_size // 2
            crop_size_h_under = 2
            crop_size_h_above = state_size - 2
            down_sample = 5

            small_board = skimage.measure.block_reduce((self._board == 0).astype(int), (down_sample, down_sample), np.max)
            w, h = np.array(small_board.shape)

            bin_board = np.zeros((w + N * 2, h + N * 2))
            bin_board[N:-N, N:-N] = small_board
            x_head, y_head = np.round(self._positions[player_id]).astype(int) // down_sample + N
            bin_board[y_head, x_head] = -20
            rotated_bord = 1 + -ndimage.rotate(bin_board, 90 - np.rad2deg(self._angles[player_id]), reshape=False)
            y_head_new, x_head_new = np.unravel_index(np.argmax(rotated_bord), rotated_bord.shape)
            rotated_bord = np.maximum(np.minimum(rotated_bord, 1), 0)

            x_right = x_head_new + crop_size_w
            x_left = x_head_new - crop_size_w
            y_top = y_head_new - crop_size_h_above
            y_bot = y_head_new + crop_size_h_under

            croped_state = rotated_bord[y_top:y_bot, x_left:x_right]

            if debug:
                fig = plt.figure(figsize=(9, 3))
                ax1, ax2, ax3 = fig.subplots(1, 3)
                ax1.imshow(self._rgb_board)
                ax1.set_title('Original Board')
                ax2.imshow(rotated_bord, cmap='gray')
                ax2.scatter(x_head_new, y_head_new)
                ax2.set_title('Rotated Board')
                ax3.imshow(croped_state, cmap='gray')
                ax3.set_title('Cropped State')
                plt.show()
            return np.expand_dims(croped_state, axis=-1)

    @classmethod
    def from_state(cls, other, *args, **kwargs):
        # return copy.deepcopy(other)
        new_state = copy.copy(other)
        new_state.margin = other.margin
        new_state._rgb_board = other._get_rgb_board().copy()
        new_state._board = other.get_board().copy()
        new_state.colors = other.colors
        new_state._positions = copy.deepcopy(other.get_all_positions())
        new_state.alive = copy.copy(other.alive)
        new_state._angles = copy.copy(other.get_all_angles())
        new_state.extract_features = copy.copy(other.extract_features)
        new_state.arena_size = other.arena_size
        return new_state

    def draw_circle(self, color_2d, color, center, radius):
        circle = CIRCLES[radius - 1]
        circle = np.array(circle) + np.array(center)
        circle = np.round(circle).astype(np.int)
        circle = self.clip(circle)
        self._rgb_board[circle[..., 1] + self.margin, circle[..., 0] + self.margin, ...] = color
        self._board[circle[..., 1], circle[..., 0]] = color_2d

    def clip(self, circle):
        circle[circle < 0] = 0
        circle[circle[..., 0] >= self.arena_size, 0] = self.arena_size - 10
        circle[circle[..., 1] >= self.arena_size, 1] = self.arena_size - 10
        return np.round(circle).astype(np.int)

    def draw_player(self, player_id, use_color=False):
        self.counts[player_id] += 1
        rgb_color = self.colors[player_id] if use_color else BLACK
        color = player_id + 2 if use_color else BLACK_2D
        try:
            self.draw_circle(color, rgb_color, self._positions[player_id], PLAYER_RADIUS)
        except:
            self.draw_circle(color, rgb_color, self._positions[player_id], PLAYER_RADIUS)

    def draw_head(self, position):
        try:
            self.draw_circle(HEAD_2D, HEAD_COLOR, position, HEAD_RADIUS)
        except:
            self.draw_circle(HEAD_2D, HEAD_COLOR, position, HEAD_RADIUS)

    def is_terminal_state(self):
        return np.sum(self.alive) < 1

    def get_distance_to_obstacles(self, initial_angle, position, num_angles, max_distance=150):
        angles = [(-np.pi / 2) + ((i / (num_angles - 1)) * np.pi) for i in range(num_angles)]
        angles = initial_angle + np.array(angles)
        distances = np.zeros_like(angles)
        for i, angle in enumerate(angles):
            distances[i] = self.distance_to_obstacle(position, angle, max_distance)
        return distances / max_distance

    def distance_to_obstacle(self, position, angle, max_distance=150):
        position = np.array(position) + self.margin
        distances = np.arange(10, max_distance)
        xx = position[0] + (np.cos(angle) * distances)
        yy = position[1] - (np.sin(angle) * distances)
        xx = np.round(xx).astype(np.int)
        yy = np.round(yy).astype(np.int)
        xx[xx < self.margin] = self.margin - 1
        yy[yy < self.margin] = self.margin - 1
        xx[xx >= self.arena_size + (self.margin * 2)] = self.arena_size + (self.margin * 2) - 1
        yy[yy >= self.arena_size + (self.margin * 2)] = self.arena_size + (self.margin * 2) - 1
        nonzero = np.nonzero(self._rgb_board[yy, xx, ...])
        if len(nonzero[0]) > 0:
            return np.linalg.norm(position - np.array([xx[nonzero[0][0]], yy[nonzero[0][0]]]))
        return max_distance

    def get_player_drl_features(self, player_id):
        features = [self._positions[player_id][0] / self.arena_size, self._positions[player_id][1] / self.arena_size]
        return features
