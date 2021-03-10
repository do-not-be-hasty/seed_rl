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
import sys

from absl import flags
from absl import logging

import gym
from seed_rl.common import common_flags
from seed_rl.common import env_wrappers
from seed_rl.starcraft import observation

from smac.env import StarCraft2Env

def create_environment(_):
  """Returns a starcraft environment."""
  task = '3s5z'

  logging.info('Creating environment: %s', task)
  env = StarCraft2Env(map_name=task)
  return env_wrappers.SCWrapper(env)
