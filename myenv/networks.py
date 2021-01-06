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

"""SEED agent using Keras."""

import collections
from seed_rl.common import utils
from seed_rl.myenv import observation
import tensorflow as tf


AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks):
    
    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same',
                                        kernel_initializer='lecun_normal')
    self._max_pool = tf.keras.layers.MaxPool2D(
        pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out


class GFootball(tf.Module):
  """Agent with ResNet, but without LSTM and additional inputs.

  Four blocks instead of three in ImpalaAtariDeep.
  """

  def __init__(self, parametric_action_distribution):
    super(GFootball, self).__init__(name='gfootball')

    # Parameters and layers for unroll.
    self._parametric_action_distribution = parametric_action_distribution

    self.flatten = tf.keras.layers.Flatten()
    self.layer0 = tf.keras.layers.Dense(16, activation='relu')
    self.layer1 = tf.keras.layers.Dense(16, activation=None)

    # Layers for _head.
    self._policy_logits = tf.keras.layers.Dense(
        self._parametric_action_distribution.param_size,
        name='policy_logits',
        kernel_initializer='lecun_normal')
    self._baseline = tf.keras.layers.Dense(
        1, name='baseline', kernel_initializer='lecun_normal')

  def initial_state(self, batch_size):
    return ()

  def _single_network_pass(self, observation):
      x = self.flatten(observation)
      x = self.layer0(x)
      x = self.layer1(x)

      return x

  def _torso(self, unused_prev_action, env_output):
    _, _, frame, _, _ = env_output
    print('frame', frame.shape)
    print('single frame', frame[:,0])

    return tf.concat([self._single_network_pass(frame[:,i]) for i in range(frame.shape[1])], axis=-1)
    # return tf.stack([self._single_network_pass(frame[:,i]) for i in range(frame.shape[1])], axis=-1)

  def _single_head(self, core_output):
    policy_logits = self._policy_logits(core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = self._parametric_action_distribution.sample(policy_logits)

    return AgentOutput(new_action, policy_logits, baseline)

  def _head(self, core_output):
    print('core_output', core_output)
    # print('core_output[0]', core_output[:,:,0])
    # print('logits', self._policy_logits)

    return self._single_head(core_output)
    # print('out:', self._single_head(core_output[:,:,0]))
    # AgentOutput(action=<tf.Tensor 'Categorical_1/sample/Reshape_2:0' shape=(1,) dtype=int64>,
    # policy_logits=<tf.Tensor 'policy_logits/BiasAdd:0' shape=(1, 7) dtype=float32>,
    # baseline=<tf.Tensor 'Squeeze:0' shape=(1,) dtype=float32>)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_actions, env_outputs, core_state, unroll=False,
               is_training=False, postprocess_action=True):
    if not unroll:
      # Add time dimension.
      prev_actions, env_outputs = tf.nest.map_structure(
          lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))

    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)

    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    if postprocess_action:
      outputs = outputs._replace(
          action=self._parametric_action_distribution.postprocess(
              outputs.action))

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))
    return utils.batch_apply(self._head, (torso_outputs,)), core_state
