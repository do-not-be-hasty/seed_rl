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


"""V-trace (IMPALA) learner for Google Research Football."""

from absl import app
from absl import flags

from seed_rl.agents.vtrace import learner
from seed_rl.common import actor
from seed_rl.common import common_flags
from seed_rl.starcraft import env
from seed_rl.starcraft import networks
from seed_rl.starcraft import visualize
import tensorflow as tf

import neptune
import neptune_tensorboard
# from mrunner.helpers.client_helper import get_configuration


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
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    get_configuration(FLAGS.mrunner_config, inject_parameters_to_FLAGS=True)
    actor.actor_loop(env.create_environment)
  elif FLAGS.run_mode == 'learner':

    get_configuration(FLAGS.mrunner_config,
                      print_diagnostics=True, with_neptune=True,
                      inject_parameters_to_FLAGS=True,
                      integrate_with_tensorboard=True
                      )
    learner.learner_loop(env.create_environment,
                         create_agent,
                         create_optimizer)
  elif FLAGS.run_mode == 'visualize':
    visualize.visualize(env.create_environment, create_agent, create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


# TODO(): include this to mrunner (and write properly)
def get_configuration(config_file,
        print_diagnostics=False, with_neptune=False,
        inject_parameters_to_gin=False, inject_parameters_to_FLAGS=False,
        nesting_prefixes=(),
        env_to_properties_regexp=".*PWD",
        integrate_with_tensorboard=False
):
  import argparse
  import datetime
  import os
  import re
  import socket
  from munch import Munch
  import cloudpickle
  import ast
  import logging
  from absl import flags
  FLAGS = flags.FLAGS

  if config_file is None:
    return {}

  # with_neptune might be also an id of an experiment
  global experiment_

  params = None
  experiment = None
  git_info = None

  # # This is here for running locally, load experiment from spec
  # if commandline_args.ex:
  #   from path import Path
  #   vars_ = {'script': str(Path(commandline_args.ex).name)}
  #   exec(open(commandline_args.ex).read(), vars_)
  #   experiments = vars_['experiments_list']
  #   logger_.info("The specifcation file contains {} "
  #                "experiments configurations. The first one will be used.".format(
  #     len(experiments)))
  #   experiment = experiments[0]
  #   params = experiment.parameters

  # # This is here for running remotely, load experiment from dump
  # if commandline_args.config:
  #   logger_.info("File to load:{}".format(commandline_args.config))
  with open(config_file, "rb") as f:
    experiment = Munch(cloudpickle.load(f))
  params = Munch(experiment['parameters'])
  git_info = experiment.get("git_info", None)
  if git_info:
    git_info.commit_date = datetime.datetime.now()

  if inject_parameters_to_gin:
    print("The parameters of the form 'aaa.bbb' will be injected to gin.")
    gin_params = {param_name: params[param_name] for param_name in params if
                  "." in param_name}
    raise NotImplementedError()

    # inject_dict_to_gin(gin_params)

  if with_neptune:
    if 'NEPTUNE_API_TOKEN' not in os.environ:
      print("Neptune will be not used.\nTo run with neptune please set your NEPTUNE_API_TOKEN variable")
    else:
      import neptune
      neptune.init(project_qualified_name=experiment.project)
      params_to_sent_to_neptune = {}
      for param_name in params:
        try:
          val = str(params[param_name])
          if val.isnumeric():
            val = ast.literal_eval(val)
          params_to_sent_to_neptune[param_name] = val
        except:
          print(
            "Not possible to send to neptune:{}. Implement __str__".format(
              param_name))

      # Set pwd property with path to experiment.
      properties = {key: os.environ[key] for key in os.environ
                    if re.match(env_to_properties_regexp, key)}
      tags = experiment.tags + [FLAGS.nonce]
      neptune.create_experiment(name=experiment.name, tags=tags,
                                params=params, properties=properties,
                                git_info=git_info)

      import atexit
      atexit.register(neptune.stop)
      experiment_ = neptune.get_experiment()
      if integrate_with_tensorboard:
        neptune_tensorboard.integrate_with_tensorflow()

  if type(with_neptune) == str:
    import neptune
    print("Connecting to experiment:", with_neptune)
    print_diagnostics = False
    neptune.init(project_qualified_name=experiment.project)
    experiment_ = neptune.project.get_experiments(with_neptune)[0]

  if print_diagnostics:
    print(
      "PYTHONPATH:{}".format(os.environ.get('PYTHONPATH', 'not_defined')))
    print("cd {}".format(os.getcwd()))
    print(socket.getfqdn())
    print("Params:{}".format(params))

  # nest_params(params, nesting_prefixes)
  # if experiment_:
  #   params['experiment_id'] = experiment_.id
  # else:
  #   params['experiment_id'] = None

  if inject_parameters_to_FLAGS:
    from absl import flags
    FLAGS = flags.FLAGS

    for p in params:
      # possibly there is a proper api for that?
      setattr(FLAGS, p, params[p])

  return params


if __name__ == '__main__':
  app.run(main)
