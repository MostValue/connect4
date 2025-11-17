# %% 
from enum import Enum

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time
import torch as t
import torch.nn.functional as F
from typing import TypeAlias
from dataclasses import dataclass


ObsType: TypeAlias = np.ndarray
ActType: TypeAlias = int

@dataclass
class GameConfig:
    """Configuration parameters for Connect Four."""
    rows: int = 6
    cols: int = 7
    x_in_row: int = 4


class Connect4Env(gym.Env):
    """
    A simple Connect 4 environment for reinforcement learning.
    The board is represented as a 6x7 grid, where players take turns dropping pieces into columns.
    The first player to connect four pieces in a row (horizontally, vertically, or diagonally) wins.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    class Player(Enum):
        EMPTY = 0
        PLAYER1 = 1 # by default, red, (X)
        PLAYER2 = 2  # by default, yellow, (O)

    def __init__(self, game_config: GameConfig = GameConfig(), render_mode=None) -> None:
        super(Connect4Env, self).__init__()

        self.ROWS = game_config.rows
        self.COLS = game_config.cols
        self.CONNECT_N = game_config.x_in_row

        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Invalid render mode: {render_mode}"
        self.render_mode = render_mode

        # Define action and observation space
        self.action_space = spaces.Discrete(self.COLS)

        # The observation is the state of the board, maximum value is 2 (PLAYER2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.ROWS, self.COLS), dtype=np.int8)

        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)

        # player 1 starts first
        self.current_player = self.Player.PLAYER1
        self.done = False

        # rendering attributes for human mode, none until initialized)
        self.window = None
        self.clock = None

    def _get_obs(self) -> ObsType:
        return self.board.copy()
    
    def _get_info(self) -> dict:
        return {"current_player": self.current_player.value}
    
    def _get_valid_actions(self) -> list[ActType]:
        # return columns that are not full
        return [c for c in range(self.COLS) if self.board[0, c] == self.Player.EMPTY.value]
    
    def _check_winner(self) -> bool:
        """Check if the current player has won the game."""

        # We apply a convolution to check whether there exists a connect 4
        player_value = self.current_player.value
        board = t.tensor(self.board == player_value, dtype=t.int8)

        kernels = [
            t.ones((self.CONNECT_N, 1), dtype=t.int8),  # vertical
            t.ones((1, self.CONNECT_N), dtype=t.int8),  # horizontal
            t.eye(self.CONNECT_N, dtype=t.int8),  # diagonal \
            t.flip(t.eye(self.CONNECT_N, dtype=t.int8), dims=[1])  # diagonal /
        ]

        for kernel in kernels:
            conv = F.conv2d(
                board.unsqueeze(0).unsqueeze(0).float(), # (1, 1, H, W)
                kernel.unsqueeze(0).unsqueeze(0).float(), # (1, 1, kH, kW)
            )
            if t.any(conv >= self.CONNECT_N):
                return True
        return False
    
    
    def _play_turn(self, action: ActType) -> None:
        """Place a piece for the current player in the selected column."""

        assert self.action_space.contains(action), "Invalid action."
        assert action in self._get_valid_actions(), "Column is full."

        for row in reversed(range(self.ROWS)):
            if self.board[row, action] == self.Player.EMPTY.value:
                self.board[row, action] = self.current_player.value
                return 


    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:

        # validity checks
        if self.done:
            raise ValueError("Game is over. Please reset the environment.")
       
        # play the turn (which checks validity internally)
        self._play_turn(action)

        if self.render_mode == "human":
            self._render_frame()
        # winner
        if self._check_winner():
            self.done = True
            reward = 1.0
            terminated = True
            truncated = False

        # draw
        elif np.all(self.board != self.Player.EMPTY.value):
            self.done = True
            reward = 0.0
            terminated = False
            truncated = True

        # game continues
        else:
            reward = 0.0
            terminated = False
            truncated = False
            # switch player
            self.current_player = (
                self.Player.PLAYER2 if self.current_player == self.Player.PLAYER1 else self.Player.PLAYER1
            )
        
        
        observation = self._get_obs()
        info = self._get_info() 
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None) -> tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.Player.PLAYER1
        self.done = False

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()
    
        return observation, info
    
    # rendering using pygame
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((700, 600))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()


        canvas = pygame.Surface((700, 600))
        canvas.fill((0, 0, 255))  # blue background 
        for c in range(self.COLS):
            for r in range(self.ROWS):
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),  # black for empty
                    (c * 100 + 50, r * 100 + 50),
                    45,
                )
                if self.board[r, c] == self.Player.PLAYER1.value:
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),  # red
                        (c * 100 + 50, r * 100 + 50),
                        45,
                    )
                elif self.board[r, c] == self.Player.PLAYER2.value:
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 0),  # yellow
                        (c * 100 + 50, r * 100 + 50),
                        45,
                    )
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0)) # copy canvas to window
            pygame.event.pump() # handle window events
            pygame.display.update() # update the display

            # enforces the framerate
            self.clock.tick(self.metadata["render_fps"])
        else:
            return t.tensor(pygame.surfarray.array3d(canvas)).permute(1, 0, 2).numpy()
    
    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

gym.register(
    id="Connect4-v0",
    entry_point=Connect4Env,
)
