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


"""VTRACE devel setup

Run me with: vtrace_main_dev.py --run_mode=actor --nonce=abc --logtostderr --pdb_post_mortem --inference_batch_size 1 --num_envs 1
"""

from absl import app
from absl import flags

from seed_rl.agents.vtrace import learner_dev as learner
from seed_rl.common import actor_dev as actor
from seed_rl.common import common_flags
from seed_rl.starcraft import env
from seed_rl.starcraft import networks
import tensorflow as tf


FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to visualize')
flags.DEFINE_string('mrunner_config', None, 'Mrunner config file.')


def create_agent(unused_action_space, unused_env_observation_space,
                 parametric_action_distribution):
  return networks.GFootball(parametric_action_distribution)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def main(argv):
  tf.config.run_functions_eagerly(True)

  actor_iterator0 = actor.actor_loop(env.create_environment)
  learner_iterator = learner.learner_loop(env.create_environment,
                                          create_agent, create_optimizer)
  next(learner_iterator)
  next(actor_iterator0)

  for i in range(11):
    next(actor_iterator0)
  next(learner_iterator)

  for i in range(11):
    next(actor_iterator0)
  next(learner_iterator)


# if __name__ == '__main__':
app.run(main)
