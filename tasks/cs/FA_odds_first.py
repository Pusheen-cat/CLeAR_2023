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

"""Odds first task for generalization."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from tasks import task


class OddsFirst(task.GeneralizationTask):
  """A task which goal is to output the tokens at odd indices of a string first.

  The input is a string s_1 ... s_n composed of symbols from a finite set S. The
  output is the same string, but where the values at odd indexes have been put
  first: s_1 s_3 s_5 ... s_2 s_4 s_6 ...

  Examples:
    00110101 -> 0100 0111
    110 -> 10 1

  In the paper, we use only binary strings (ie S = {0, 1}).
  Note that the sampling is jittable so this task is fast.
  """

  def __init__(self, fa_dim:int, vocab_size: int=2, *args, **kwargs):
    """Initializes the odds_first task.

    Args:
      vocab_size: The size of the alphabet.
      *args: Args for the base task class.
      **kwargs: Kwargs for the base task class.
    """
    super().__init__(*args, **kwargs)
    self.fa_dim = fa_dim
    self._vocab_size = vocab_size

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and their outputs."""
    rng, rng_trans = jax.random.split(rng)
    strings = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=self._vocab_size)
    one_hot_strings = jnn.one_hot(strings, num_classes=self._vocab_size)
    output = jnp.concatenate(
        [one_hot_strings[:, 1::2], one_hot_strings[:, ::2]], axis=1)
    one_hot_strings = self.fa_transform(rng_trans, strings, length)
    return {"input": one_hot_strings, "output": output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the model."""
    return self.fa_dim

  @property
  def output_size(self) -> int:
    """Returns the output size for the model."""
    return self._vocab_size

  def output_length(self, input_length: int) -> int:
    """Returns the output length for the model."""
    return input_length

  def fa_transform(self, rng, x, length):
    # 2 --> 1 features
    #return jnp.expand_dims(x, 2)
    pos, rng= jax.random.split(rng)
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

