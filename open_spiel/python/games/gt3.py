# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Generalized version of Tic tac toe (noughts and crosses), implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""

from typing import Mapping, Optional

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {
  "board_size": 3,
  "win_condition": 3
}
_GAME_TYPE = pyspiel.GameType(
    short_name="python_gt3",
    long_name="Python GT3",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS)


class GT3Game(pyspiel.Game):
  """A Python version of a Generalized Tic-Tac-Toe game."""

  def __init__(
      self,
      params: Optional[Mapping[str, int]] = None
  ):
    self.board_size = (
      params["board_size"]
      if params else _DEFAULT_PARAMS["board_size"])
    self.num_cells = self.board_size * self.board_size
    self.win_condition = (
      params["win_condition"]
      if params else _DEFAULT_PARAMS["win_condition"])
    game_info = pyspiel.GameInfo(
      num_distinct_actions=self.num_cells,
      max_chance_outcomes=0,
      num_players=2,
      min_utility=-1.0,
      max_utility=1.0,
      utility_sum=0.0,
      max_game_length=self.num_cells
    )
    super().__init__(_GAME_TYPE, game_info, params if params else _DEFAULT_PARAMS)

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return GT3State(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(self, params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class GT3State(pyspiel.State):
  """A python version of the Generalized Tic-Tac-Toe state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board_size = game.board_size
    self.board = np.full((self.board_size, self.board_size), ".")
    self.win_condition = game.win_condition

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    return [a for a in range(self.board_size * self.board_size) if self.board[_coord(self.board_size, a)] == "."]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    self.board[_coord(self.board_size, action)] = "x" if self._cur_player == 0 else "o"
    if _win_exists(self):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    row, col = _coord(self.board_size, action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, game, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (1 + _NUM_PLAYERS, game.board_size, game.board_size)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs.fill(0)
    for row in range(state.board_size):
      for col in range(state.board_size):
        cell_state = ".ox".index(state.board[row, col])
        obs[cell_state, row, col] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)


# Helper functions for game details.


def _win_exists(state):
  """Checks if win condition exists, returns "x" or "o" if so, and None otherwise."""
  # recall 'win_condition' is num in a row to connect
  player = 'o' if state.current_player() else 'x'
  # Check rows
  for row in range(state.board_size):
      for col in range(state.board_size - state.win_condition + 1):
          win_check = 0
          if state.board[row][col] == player:
              current_col = col
              while current_col < state.board_size and state.board[row][current_col] == player:
                  win_check += 1
                  current_col += 1
              if win_check >= state.win_condition:
                  return player

  # Check columns
  for col in range(state.board_size):
      for row in range(state.board_size - state.win_condition + 1):
          win_check = 0
          if state.board[row][col] == player:
              current_row = row
              while current_row < state.board_size and state.board[current_row][col] == player:
                  win_check += 1
                  current_row += 1
              if win_check >= state.win_condition:
                  return player

  # Check L-R diagonal
  for row in range(state.board_size - state.win_condition + 1):
      for col in range(state.board_size - state.win_condition + 1):
          win_check = 0
          if state.board[row][col] == player:
              current_row = row
              current_col = col
              while current_row < state.board_size and current_col < state.board_size \
                      and state.board[current_row][current_col] == player:
                  win_check += 1
                  current_row += 1
                  current_col += 1
          if win_check >= state.win_condition:
              return player

  # Check R-L diagonal
  for row in range(state.board_size - state.win_condition + 1):
      for col in range(state.board_size - 1, state.win_condition - 2, -1):
          win_check = 0
          if state.board[row][col] == player:
              current_row = row
              current_col = col
              while current_row < state.board_size and current_col >= 0 \
                      and state.board[current_row][current_col] == player:
                  win_check += 1
                  current_row += 1
                  current_col -= 1
          if win_check >= state.win_condition:
              return player


def _coord(board_size, move):
  """Returns (row, col) from an action id."""
  return (move // board_size, move % board_size)


def _board_to_string(board):
  """Returns a string representation of the board."""
  return "\n".join("".join(row) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, GT3Game)
