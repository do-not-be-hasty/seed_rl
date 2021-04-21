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


"""Flags common across learners and actors.
"""

from absl import flags
import numpy as np

flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_alias('job-dir', 'logdir')
flags.DEFINE_string('server_address', 'localhost:8686', 'Server address.',
                    allow_hide_cpp=True)
flags.DEFINE_string('replay_dir', '/tmp/agent/replays',
                    'Directory for recorded replays.')

flags.DEFINE_string('nonce', None, 'Unique ID of experiment')

flags.DEFINE_enum(
    'run_mode', None, ['learner', 'actor', 'visualize'],
    'Whether we run the learner or the actor. Each actor runs the environment '
    'and sends to the learner each env observation and receives the action to '
    'play. A learner performs policy inference for batches of observations '
    'coming from multiple actors, and use the generated trajectories to learn.')

flags.DEFINE_integer('num_eval_envs', 0,
                     'Number of environments that will be used for eval '
                     ' (for agents that support eval environments).')
flags.DEFINE_integer(
    'env_batch_size', 1,
    'How many environments to operate on together in a batch.'
)
flags.DEFINE_integer('num_envs', 4,
                     'Total number of environments in all actors.')
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')

flags.DEFINE_bool('is_local', False,
                  'Whether the program is running locally.')
flags.DEFINE_bool('is_centralized', True,
                  'Whether to use centralized critic network.')
flags.DEFINE_bool('is_action_aware', False,
                  'Use action aware centralized training')
flags.DEFINE_bool('centralized_IS', False,
                  'Use centralized importance sampling')
flags.DEFINE_bool('mean_value_function', False,
                  'Use mean value function in the decentralized training')
