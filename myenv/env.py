# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Football env factory."""

from absl import flags
from absl import logging

import gym
from seed_rl.common import common_flags
from seed_rl.common import env_wrappers
from seed_rl.myenv import observation
import marlgrid
from marlgrid import envs


def create_environment(_):
  """Returns a gym Football environment."""
  task = 'MarlGrid-3AgentCluttered11x11-v0'
  # task = 'LunarLander-v2'
  logging.info('Creating environment: %s', task)
  env = gym.make(task)
  return env_wrappers.MultiWrapper(env, num_agents=3)
  # return env_wrappers.FloatWrapper(env)


# import marlgrid
# from marlgrid import envs
#
# def tmp():
#   env = gym.make("MarlGrid-3AgentCluttered11x11-v0")
#   # obs: [obs1, obs2, obs3]
#   # action: [act1, act2, act3]
#   #
#   # >>> env.observation_space
#   # Tuple(Box(0, 255, (56, 56, 3), uint8), Box(0, 255, (56, 56, 3), uint8), Box(0, 255, (56, 56, 3), uint8))
#   #
#   # >>> env.action_space
#   # Tuple(Discrete(7), Discrete(7), Discrete(7))
