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

"""Manipulate an input stack, using the input actions."""

from typing import List, Mapping, Tuple

import chex
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from tasks import task


class StackManipulation(task.GeneralizationTask):
  """A task with the goal of following instructions and returning the end stack.

  The input is composed of a stack of 0s and 1s followed by a sequence of
  instructions POP/PUSH 0/PUSH 1 (represented by 2s/3s/4s). The input stack is
  given bottom-to-top, and the agent needs to execute the instructions given
  (left-to-rigth) and output the final stack top-to-bottom (i.e., as if it were
  popping the final stack). If a POP action is to be called on an empty stack,
  the action is ignored. The output is padded with 0s to match the input length
  + 1 (to accommodate for the termination token), and the end of the final stack
  is denoted with the termination symbol 2 (i.e., the output has values in {0,
  1, 2}).

  Examples:
    0 1 1 0 PUSH 1 POP POP
      initial 0 1 1 0       (the stack is received bottom-to-top)
      PUSH 1  0 1 1 0 1
      POP     0 1 1 0
      POP     0 1 1
    -> 1 1 0 2 0 0 0 0      (the stack is returned top-to-bottom)

    1 1 0 POP POP POP
      initial 1 1 0
      POP     1 1
      POP     1
      POP
    -> 2 0 0 0 0 0 0 0      (the stack is empty and padded with zeros)
  """
  def __init__(self, fa_dim=4):
    self.fa_dim = fa_dim
    if fa_dim !=4:
        raise Exception("Stack_manipulation need additional column for position")

  def _sample_expression_and_result(
      self, length: int) -> Tuple[np.ndarray, List[int]]:
    """Returns an expression with stack instructions, and the result stack."""
    if length == 1:
      value = np.random.randint(low=0, high=2, size=(1,))
      return value, list(value), length

    # Initialize the stack content and the actions (POP/PUSH).
    stack_length = np.random.randint(low=1, high=length)
    stack = np.random.randint(low=0, high=2, size=(stack_length,))
    actions = np.random.randint(low=2, high=5, size=(length - stack_length,))

    # Apply the actions on the stack.
    current_stack = list(stack)

    for action in actions:
      if action == 2:  # POP
        if current_stack:
          current_stack.pop()
      elif action in [3, 4]:  # PUSH a 0 (case 3) or a 1 (case 4)
        current_stack.append(action - 3)

    return np.concatenate([stack, actions]), current_stack[::-1], stack_length

  def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
                   length: int) -> Mapping[str, chex.Array]:
    """Returns a batch of strings and the expected class."""
    rng, rng_trans = jax.random.split(rng)
    expressions, results, lengths = [], [], []
    for _ in range(batch_size):
      expression, result, _length = self._sample_expression_and_result(length)
      expressions.append(expression)
      # Append the termination token to the result.
      result += [self.output_size - 1]
      # Pad the result with zeros to match the input length (accounting for the
      # termination token).
      result += [0] * (length + 1 - len(result))
      results.append(result)
      lengths.append(_length)
    expressions = jnp.array(expressions)
    results = jnp.array(results)

    #inputs = jnn.one_hot(expressions, self.input_size)
    inputs = self.fa_transform(rng_trans, expressions, lengths, length)
    output = jnn.one_hot(results, self.output_size)
    return {'input': inputs, 'output': output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models.

    The value is 5 because we have two possible tokens in the stack (0, 1), plus
    three tokens to describe the PUSH 0, PUSH 1, and POP actions.
    """
    return self.fa_dim

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 3

  def output_length(self, input_length: int) -> int:
    """Returns the output length of the task."""
    return input_length + 1

  def accuracy_mask(self, target: chex.Array) -> chex.Array:
    """Computes mask that ignores everything after the termination tokens.

    Args:
      target: Target tokens of shape `(batch_size, output_length, output_size)`.

    Returns:
      The mask of shape `(batch_size, output_length)`.
    """
    batch_size, length, _ = target.shape
    termination_indices = jnp.argmax(
        jnp.argmax(target, axis=-1),
        axis=-1,
        keepdims=True,
    )
    indices = jnp.tile(jnp.arange(length), (batch_size, 1))
    return indices <= termination_indices


  def fa_transform(self, rng, x, lengths, length):
    # 2 --> 1 features
    #return jnp.expand_dims(x, 2)
    rng1, rng2, rng3= jax.random.split(rng,3)
    rnd_4 =  jax.random.randint(rng1, shape=x.shape, minval=0, maxval=4)
    rnd_3 = jax.random.randint(rng2, shape=x.shape, minval=0, maxval=3)
    rnd_2 = jax.random.randint(rng3, shape=x.shape, minval=0, maxval=2)

    # 2 --> 3 features
    x = jnp.where(x<2, x*4+rnd_4, jnp.where(x<3, 6+rnd_2, jnp.where(x==3, rnd_3, 3+rnd_3)))


    x_1 = x % 2
    x = x // 2
    x_2 = x % 2
    x = x // 2
    x_3 = x % 2

    stripe = jnp.zeros(shape=x.shape)
    if length >1:
        stripe = stripe.at[jnp.arange(x.shape[0]),lengths].set(1)

    x = jnp.stack((stripe, x_1, x_2, x_3), axis=2)

    return x
