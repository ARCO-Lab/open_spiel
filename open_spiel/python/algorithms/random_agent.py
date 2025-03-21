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

"""RL agent following an uniform distribution over legal actions."""

import numpy as np
from open_spiel.python import rl_agent


class RandomAgent(rl_agent.AbstractAgent):
  """Random agent class."""

  def __init__(self, player_id, num_actions, name="random_agent"):
    assert num_actions > 0
    self.player_id = player_id
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    # Pick a random legal action.
    cur_legal_actions = time_step.observations["legal_actions"][self.player_id] if time_step.observations["legal_actions"][self.player_id] else time_step.observations["legal_actions"][1 - self.player_id]
    action = np.random.choice(cur_legal_actions)
    probs = np.zeros(self._num_actions)
    probs[cur_legal_actions] = 1.0 / len(cur_legal_actions)

    return rl_agent.StepOutput(action=action, probs=probs)
