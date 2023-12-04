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

"""Compute the floor of the square root of a binary number."""

import math
import random

import chex
import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom

from tasks import task
from tasks.cs import binary_addition


class ComputeSqrt(task.GeneralizationTask):
  """A task which goal is to compute the square root of a binary number.

  The input is a number in binary (big-endian), and the output is the floor of
  the square root of this number, also in binary.
  Note the output length ie the length of the square root in binary is always
  ceil(input_length / 2) (because log(sqrt(x)) = 1/2 log(x)).

  Examples:
   100101 = 37 -> square root is 6.08... -> floor(6.08) = 6 -> 101
   111 = 7 -> square root is 2.64 -> floor(2.64) = 2 -> 10
  """
  def __init__(self, fa_dim=3):
    self.fa_dim = fa_dim

  def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of binary numbers and their square roots, in binary."""

    numbers = [random.randint(1, 2**length - 1) for _ in range(batch_size)]
    binary_numbers = binary_addition.numbers_to_fixed_length_binary(
        numbers, length=length, little_endian=False)

    sqrts = list(map(math.isqrt, numbers))
    binary_sqrts = binary_addition.numbers_to_fixed_length_binary(
        sqrts, length=self.output_length(length), little_endian=False)

    binary_numbers = jnp.array(binary_numbers, jnp.int32)
    binary_sqrts = jnp.array(binary_sqrts, jnp.int32)

    #inputs = jnn.one_hot(binary_numbers, self.input_size)
    inputs = self.fa_transform(rng, binary_numbers, length)
    output = jnn.one_hot(binary_sqrts, self.output_size)
    return {'input': inputs, 'output': output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self.fa_dim

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2

  def output_length(self, input_length: int) -> int:
    return math.ceil(input_length / 2)

  def fa_transform(self, rng, x, length):
    pos, rng= jax.random.split(rng)
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