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

"""Bucket sort task for generalization."""

import functools
from typing import Mapping

import chex
import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom

from tasks import task


class BucketSort(task.GeneralizationTask):
  """A task which goal is to sort tokens from a fixed alphabet.

  The input string is composed of tokens from a fixed-size alphabet, i.e.,
  `{0, 1, ..., vocab_size - 1}`, and the goal is to return the sorted string (in
  lexicographically increasing order).

  Examples:
    10204112  ->  00111224  (with `vocab_size = 5`)
    1110001   ->  0001111   (with `vocab_size = 2`)
  """

  def __init__(self, fa_dim: int=3, *args, vocab_size: int = 4, **kwargs) -> None:
    """Initializes the task.

    Args:
      *args: The args for the base task class.
      vocab_size: The size of the alphabet.
      **kwargs: The kwargs for the base task class.
    """
    super().__init__(*args, **kwargs)
    self._vocab_size = vocab_size
    self.fa_dim = fa_dim

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      length: int,
  ) -> Mapping[str, chex.Array]:
    """Returns a batch of strings and tokens sorted by (inc.) occurrence."""
    rng, rng_trans = jax.random.split(rng)
    strings = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=self._vocab_size)
    sorted_strings = jnp.sort(strings, axis=-1)

    one_hot_strings = self.fa_transform(rng_trans, strings, length)
    return {
        'input': one_hot_strings,
        'output': jnn.one_hot(sorted_strings, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self.fa_dim

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._vocab_size

  def output_length(self, input_length: int) -> int:
    """Returns the output length for a given input length."""
    return input_length

  def fa_transform(self, rng, x, length):
    # 2 --> 1 features
    #return jnp.expand_dims(x, 2)
    pos, rng= jax.random.split(rng)
    # 2 --> 3 features
    x = x*2+ jrandom.randint(rng, shape=x.shape, minval=0, maxval=2)
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
