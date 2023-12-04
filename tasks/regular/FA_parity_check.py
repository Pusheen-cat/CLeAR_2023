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

"""Compute whether the number of 1s in a string is even."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from tasks import task


class ParityCheck(task.GeneralizationTask):
  """A task which goal is to count the number of '1' in a string, modulo 2.

  The input is a string, composed of 0s and 1s. If the result is even, the class
  is 0, otherwise it's 1.

  Examples:
    1010100 -> 3 1s (odd) -> class 1
    01111 -> 4 1s (even) -> class 0

  Note that the sampling is jittable so this task is fast.
  """
  def __init__(self, fa_dim=3):
    self.fa_dim = fa_dim

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    rng, rng_trans = jax.random.split(rng)
    strings = jrandom.randint(
      rng, # key (Union[Array, PRNGKeyArray]) – a PRNG key used as the random key.
      shape=(batch_size, length),
      minval=0,
      maxval=2
    )
    n_b = jnp.sum(strings, axis=1) % 2
    n_b = jnn.one_hot(n_b, num_classes=2)
    #one_hot_strings = jnn.one_hot(strings, num_classes=2)
    one_hot_strings = self.fa_transform(rng_trans, strings, length)
    return {"input": one_hot_strings, "output": n_b}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self.fa_dim

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2

  def fa_transform(self, rng, x, length):
    pos, rng = jax.random.split(rng)

    # 2 --> 1 features
    #return jnp.expand_dims(x, 2)

    # 2 --> 3 features
    x = x*4+ jrandom.randint(rng, shape=x.shape, minval=0, maxval=4)
    x_1 = x % 2
    x = x // 2
    x_2 = x % 2
    x = x // 2
    x_3 = x % 2
    if self.fa_dim == 4:
      stripe = jnp.zeros(shape=x.shape)
      if length > 1:
        position = jrandom.randint(pos, shape=([x.shape[0]]), minval=1, maxval=length)
        stripe = stripe.at[jnp.arange(x.shape[0]), position].set(1)
      x = jnp.stack((stripe, x_1, x_2, x_3), axis=2)
    else:
      x = jnp.stack((x_1, x_2, x_3), axis=2)
    return x
