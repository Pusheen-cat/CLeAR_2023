# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute the final state after randomly walking on a circle."""

import functools
from typing import Mapping

import chex
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from tasks import task


class CycleNavigation(task.GeneralizationTask):
  """A task which goal is to compute the final state on a circle.

  The input is a string of actions, composed of 0s, 1s or -1s. The actions give
  directions to take on a finite length circle (0 is for stay, 1 is for right,
  -1 is for left). The goal is to give the final position on the circle after
  all the actions have been taken. The agent starts at position 0.

  By default, the length the circle is 5.

  Examples:
    1 -1 0 -1 -1 -> -2 = class 3
    1 1 1 -1 -> 2 = class 2

  Note that the sampling is jittable so it is fast.
  """

  def __init__(self, fa_dim=3, modulus: int = 5,):
    self.fa_dim = fa_dim
    self.modulus = modulus
    print('modulus of cycle naviagtion>>>>>>>>', modulus)

  @property
  def _cycle_length(self) -> int:
    """Returns the cycle length, number of possible states."""
    return self.modulus

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
                   length: int) -> Mapping[str, chex.Array]:
    """Returns a batch of strings and the expected class."""
    rng, rng_trans = jax.random.split(rng, num=2)
    actions = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=3)
    final_states = jnp.sum(actions - 1, axis=1) % self._cycle_length
    final_states = jnn.one_hot(final_states, num_classes=self.output_size)
    one_hot_strings, debug = self.fa_transform(rng_trans, actions, length)
    #one_hot_strings = jnn.one_hot(actions, num_classes=self.input_size)
    return {"input": one_hot_strings, "output": final_states, 'debug':debug}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self.fa_dim

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._cycle_length

  def fa_transform(self, rng, x, length):
    debug = []
    debug.append(x)
    # 3 --> 8 features
    rng1, rng2, pos = jax.random.split(rng, 3)
    rand_add_2 = jrandom.randint(rng1, shape=x.shape, minval=0, maxval=2)
    rand_add_3 = jrandom.randint(rng2, shape=x.shape, minval=0, maxval=3)
    x = jnp.where(x<2, x*3+rand_add_3, 6+rand_add_2)

    debug.append(x)

    #return jnn.one_hot(x, num_classes=self.fa_dim) # if fa_dim=8
    x_1 = x%2
    x = x//2
    x_2 = x%2
    x = x //2
    x_3 = x%2
    if self.fa_dim == 4:
      stripe = jnp.zeros(shape=x.shape)
      if length > 1:
        position = jrandom.randint(pos, shape=([x.shape[0]]), minval=1, maxval=length)
        stripe = stripe.at[jnp.arange(x.shape[0]), position].set(1)
      x = jnp.stack((stripe, x_1, x_2, x_3), axis=2)
    else:
      x = jnp.stack((x_1, x_2, x_3), axis=2)
    return x, debug






