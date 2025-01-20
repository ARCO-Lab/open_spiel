"""Generalized Tic-Tac-Toe agent trainer

This file is used to train your agent against itself.
You can modify the game settings & number of training episodes"""


# Change board size, win condition, and number of training episodes here
BOARD_SIZE = 3
WIN_CONDITION = 3
NUM_EPISODES = 3000


import logging
import sys
from absl import app
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.games import gt3
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner


def pretty_board(time_step):
  """Returns the board in `time_step` in a human readable format."""
  info_state = time_step.observations["info_state"][0]
  x_locations = np.nonzero(info_state[BOARD_SIZE*BOARD_SIZE:BOARD_SIZE*BOARD_SIZE*2])[0]
  o_locations = np.nonzero(info_state[BOARD_SIZE*BOARD_SIZE*2:])[0]
  board = np.full(BOARD_SIZE * BOARD_SIZE, ".")
  board[x_locations] = "X"
  board[o_locations] = "0"
  board = np.reshape(board, (BOARD_SIZE, BOARD_SIZE))
  return board


def command_line_action(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1
  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue
  return action


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  wins = np.zeros(2)
  for player_pos in range(2):
    if player_pos == 0:
      cur_agents = [trained_agents[0], random_agents[1]]
    else:
      cur_agents = [random_agents[0], trained_agents[1]]
    for _ in range(num_episodes):
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
      if time_step.rewards[player_pos] > 0:
        wins[player_pos] += 1
  return wins / num_episodes


def main(_):
  game = gt3.GT3Game({'board_size': BOARD_SIZE, 'win_condition': WIN_CONDITION})
  num_players = 2

  env = rl_environment.Environment(game)
  num_actions = env.action_spec()["num_actions"]

  agents = [
      tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  # 1. Train the agents
  for cur_episode in range(NUM_EPISODES):
    if cur_episode % int(1e4) == 0:
      win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
      logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      time_step = env.step([agent_output.action])

    # Episode is over, step all agents with final info state.
    for agent in agents:
      agent.step(time_step)


  # 2. Play from the command line against the trained agent.
  human_player = 1
  while True:
    logging.info("You are playing as %s", "O" if human_player else "X")
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      if player_id == human_player:
        agent_out = agents[human_player].step(time_step, is_evaluation=True)
        logging.info("\n%s", agent_out.probs.reshape((BOARD_SIZE, BOARD_SIZE)))
        logging.info("\n%s", pretty_board(time_step))
        action = command_line_action(time_step)
      else:
        agent_out = agents[1 - human_player].step(time_step, is_evaluation=True)
        action = agent_out.action
      time_step = env.step([action])

    logging.info("\n%s", pretty_board(time_step))

    logging.info("End of game!")
    if time_step.rewards[human_player] > 0:
      logging.info("You win")
    elif time_step.rewards[human_player] < 0:
      logging.info("You lose")
    else:
      logging.info("Draw")
    # Switch order of players
    human_player = 1 - human_player


if __name__ == "__main__":
  app.run(main)

