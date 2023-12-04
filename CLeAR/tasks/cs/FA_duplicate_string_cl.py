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

"""Duplicate string task for generalization."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from tasks import task


class DuplicateString(task.GeneralizationTask):
  """A task which goal is to duplicate a string.

  The input is a string s_1 ... s_n composed of symbols from a finite set S. The
  output is the same string outputted twice without any separator, ie:
  s_1 ... s_n s_1 ... s_n

  Examples:
    101 -> 101 101
    111111 -> 111111 111111

  In the paper, we use only binary strings (ie S = {0, 1}).
  Note that the sampling is jittable so this task is fast.
  """

  def __init__(self, fa_dim:int =3, vocab_size: int =2, *args, **kwargs):
    """Initializes the remember_string task.

    Args:
      vocab_size: The size of the alphabet.
      *args: Args for the base task class.
      **kwargs: Kwargs for the base task class.
    """
    super().__init__(*args, **kwargs)
    self.fa_dim = fa_dim
    self._vocab_size = vocab_size
    print('Vocab size of Duplicate/Reverse string >>>>>>>>>>', vocab_size)

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and their copies."""
    rng, rng_trans = jax.random.split(rng)
    strings = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=self._vocab_size)
    one_hot_strings = jnn.one_hot(strings, num_classes=self._vocab_size)
    output = jnp.concatenate([one_hot_strings, one_hot_strings], axis=1)
    mapped = self.fa_transform(rng_trans, strings, length)
    return {"input": mapped, "output": output, "original":one_hot_strings}

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
    return 2 * input_length

  def fa_transform(self, rng, x, length):

    rng1, rng2, rng3, pos = jax.random.split(rng, 4)
    rand_add_2 = jrandom.randint(rng1, shape=x.shape, minval=0, maxval=2)
    rand_add_3 = jrandom.randint(rng2, shape=x.shape, minval=0, maxval=3)
    rand_add_4 = jrandom.randint(rng3, shape=x.shape, minval=0, maxval=4)
    if self._vocab_size == 2:
        tmp = x * 4 + rand_add_4
    if self._vocab_size == 3:
        tmp = jnp.where(x < 2, x * 3 + rand_add_3, 6 + rand_add_2)
    if self._vocab_size == 4:
        tmp = x * 2 + rand_add_2
    if self._vocab_size == 5:
        tmp = jnp.where(x < 3, x * 2 + rand_add_2, x + 3)
    if self._vocab_size == 6:
        tmp = jnp.where(x < 2, x * 2 + rand_add_2, x + 2)
    if self._vocab_size == 7:
        tmp = jnp.where(x == 0, rand_add_2, x + 1)
    if self._vocab_size == 8:
        tmp = x
    x= tmp
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