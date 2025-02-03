'''
Run this file to train your DQN agent on the game of GT3.
The trained agent will be saved to my_models/trained_models
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from open_spiel.python.games import gt3
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch import my_dqn
from open_spiel.python.utils.replay_buffer import ReplayBuffer

# set path where the trained model will be saved
OUTPUT_MODEL_PATH = 'my_models/trained_models/trained_fc.pth'
OUTPUT_PLOT_PATH = 'my_models/trained_models/training_fc.png'

# define game parameters
BOARD_SIZE = 3
WIN_CONDITION = 3

# define training hyperparameters
EPOCHS = int(5e4)
REPLAY_BUFFER_CAPACITY = 10000
BATCH_SIZE = 128                    # BATCH_SIZE must be less than or equal to REPLAY_BUFFER_CAPACITY
REPLAY_BUFFER_CLASS = ReplayBuffer
LEARNING_RATE = 0.001
UPDATE_TARGET_NETWORK_EVERY = 1000
LEARN_EVERY = 10
DISCOUNT_FACTOR = 1.0
MIN_BUFFER_SIZE_TO_LEARN = 1000     # MIN_BUFFER_SIZE_TO_LEARN must be less than or equal to REPLAY_BUFFER_CAPACITY
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_DURATION = int(1e5)
OPTIMIZER_STR = "sgd"               # OPTIMIZER_STR options: "adam", "sgd"
LOSS_STR = "mse"                    # LOSS_STR options: "mse", "huber"

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes

if __name__ == "__main__":

    # use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Training using {device}...")

    game = gt3.GT3Game({"board_size": BOARD_SIZE, "win_condition": WIN_CONDITION})
    num_players = 2
    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # initialize DQN agents
    agents = [
        my_dqn.DQN(
            player_id=idx,
            state_representation_size=state_size,
            num_actions=num_actions,
            replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            replay_buffer_class=REPLAY_BUFFER_CLASS,
            learning_rate=LEARNING_RATE,
            update_target_network_every=UPDATE_TARGET_NETWORK_EVERY,
            learn_every=LEARN_EVERY,
            discount_factor=DISCOUNT_FACTOR,
            min_buffer_size_to_learn=MIN_BUFFER_SIZE_TO_LEARN,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_duration=EPSILON_DECAY_DURATION,
            optimizer_str=OPTIMIZER_STR,
            loss_str=LOSS_STR,
            accelerator=device
        )
        for idx in range(num_players)
    ]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # initialize a dictionary to store training history
    H = {
        "loss": [],
        "mean reward for agent 0": [],
        "mean reward for agent 1": [],
    }

    # train agent
    for ep in range(EPOCHS):
        if ep and ep % 1000 == 0:
            print(f"[{ep}/{EPOCHS}] DQN loss {agents[0].loss:.3f}")

            r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
            print(f"Mean epsiode rewards: {r_mean}")

            H["loss"].append(agents[0].loss.item())
            H["mean reward for agent 0"].append(r_mean[0])
            H["mean reward for agent 1"].append(r_mean[1])

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)


        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    # save the trained model
    agents[0].save(OUTPUT_MODEL_PATH)


    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["loss"], label="loss")
    plt.plot(H["mean reward for agent 0"], label="mean reward for agent 0")
    plt.plot(H["mean reward for agent 1"], label="mean reward for agent 1")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(OUTPUT_PLOT_PATH)
