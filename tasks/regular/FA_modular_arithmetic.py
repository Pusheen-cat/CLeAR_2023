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

"""Simple version of the modular arithmetic task.

Note this one allows to generate samples using a jittable function, and is
therefore much faster than its 'brackets' counterpart, which requires to
simulate the full CF grammar, non-jittable.
"""

import functools
from typing import Mapping, Optional, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from tasks import task

# Public as this may be used to encode/decode strings of numbers/symbols.
OP_BY_CHARACTER = {'+': 0, '-': 1, '*': 2, '_': 3}


def _replace_subtractions(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
  """Replaces subtractions in an expression by additions with the inverse.

  e.g. the expression [1, -, 3] results in [1, +, -3].

  Args:
    expression: Encoded expression (a 1D array of integers) in which to replace
      subtractions.
    modulus: The modulus to use for the modular arithmetic.

  Returns:
    The expression with all subtractions replaced by additions with the inverse.
  """
  if expression.size < 2:
    return expression

  mask = (expression == modulus + OP_BY_CHARACTER['-'])
  subtract_replaced = jnp.where(mask, modulus + OP_BY_CHARACTER['+'],
                                expression)
  return subtract_replaced.at[2:].multiply(1 - 2 * mask[1:-1])


def _perform_multiplications(expression: jnp.ndarray,
                             modulus: int) -> jnp.ndarray:
  """Performs all multiplications in an expression containing only + and *.

  This is done at fixed length and the result is zero-padded to achieve this.
  Since the result of performing multiplications is an expression containing
  only + operators, the operators are dropped from the output. For example, the
  expression [1, +, 3, *, 4] results in [1, 12, 0].

  Args:
    expression: Encoded expression in which to perform multiplications.
    modulus: The modulus to use for the modular arithmetic.

  Returns:
    An array with the results of the multiplications (potentially zero-padded).
  """
  term_ids = jnp.cumsum(expression == modulus + OP_BY_CHARACTER['+'])[::2]
  # Segment_prod can only be jit-compiled with a fixed number of segments.
  # Therefore, we have to set to the maximum number of terms possible and
  # mask out superfluous segment results with zeros afterwards.
  maximum_term_number = expression.shape[0] // 2 + 1
  products = jax.ops.segment_prod(
      expression[::2],
      term_ids,
      num_segments=maximum_term_number,
      indices_are_sorted=True)
  valid_segment_mask = jnp.arange(maximum_term_number) <= term_ids[-1]
  return products * valid_segment_mask


def _replace_blanks(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
  """Replaces blank symbols in expression with either `+` or `0`.

  Depending on whether the blank symbol is at the position of an operator or a
  residual, the blank symbol is replaced with a `+` operator or a `0`.

  Args:
    expression: Encoded expression in which to replace blank symbols.
    modulus: The modulus to use for the modular arithmetic.

  Returns:
    An array with blank symbols replaced by either `+` or `0`.
  """
  mask = (expression == OP_BY_CHARACTER['_'] + modulus)
  operator_mask = mask.at[::2].set(False)
  residual_mask = mask.at[1::2].set(False)

  blanks_replaced = jnp.where(operator_mask, OP_BY_CHARACTER['+'] + modulus,
                              expression)
  blanks_replaced = jnp.where(residual_mask, 0, blanks_replaced)
  return blanks_replaced


def _evaluate_expression(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
  """Returns the result of evaluating a modular arithmetic expression."""
  expression = _replace_blanks(expression, modulus)
  expression = _replace_subtractions(expression, modulus)
  additive_terms = _perform_multiplications(expression, modulus)
  return jnp.sum(additive_terms) % modulus


class ModularArithmetic(task.GeneralizationTask):
  """A task whose goal is to reduce a simple arithmetic expression.

  The input is a string, composed of numbers (in {0, ..., modulus-1}), and
  operators (in {+, -, *}). The output is the reduced value of this expression,
  which is also in {0, ..., modulus-1}.

  Examples (modulo 5):
    1 + 2 * 3 = 2
    1 - 1 - 1 = 4
    0 * 1 + 4 * 3 - 2 = 0

  Note that the input strings are always of odd length.
  """

  def __init__(self,
               fa_dim=3,
               modulus: int = 5,
               *args,
               operators: Optional[Sequence[str]] = None,
               **kwargs):
    """Initializes the modular arithmetic task.

    Args:
      modulus: The modulus used for the computation.
      *args: Args for the base task class.
      operators: Operators to be used in the sequences. By default it's None,
        meaning all operators available are used.
      **kwargs: Kwargs for the base task class.
    """
    super().__init__(*args, **kwargs)
    self.fa_dim = fa_dim
    self._modulus = modulus
    print('operator', operators)
    if operators is None:
      operators = ('+', '*', '-')
    self._operators = (OP_BY_CHARACTER[op] for op in operators) #! OP_BY_CHARACTER = {'+': 0, '-': 1, '*': 2, '_': 3}
    self._operators = list(self._operators) #[0,2,1]

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(
      self,
      rng: jnp.ndarray,
      batch_size: int,
      #!sequence_length: int,
      length: int, #! new line
  ) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of modular arithmetic expressions and their labels.

    Args:
      rng: The jax random number generator.
      batch_size: The size of the batch returned.
      sequence_length: The length of the sequence. As this length must be odd
        for the modular arithmetic dataset, if it's not, we force it to be by
        subtracting one to the length passed.
    """

    sequence_length = length
    eff_length = length
    # Subtracting one to the length if it's not odd already.
    if sequence_length % 2 != 1:
      eff_length -= 1
      #sequence_length -= 1
    rng1, rng2, rng_trans = jax.random.split(rng, num=3)


    batch = jnp.empty((batch_size, sequence_length), dtype=int)
    
    remainders = jax.random.randint(rng1,
                                    (batch_size, (sequence_length+1) // 2 ), 0,
                                    self._modulus) # _modulus: 5

    ops = self._modulus + jnp.array(list(self._operators)) #[5,7,6]

    operations = jrandom.choice(rng2, ops, (batch_size, sequence_length // 2))
    batch = batch.at[:, ::2].set(remainders)
    expressions = batch.at[:, 1::2].set(operations)

    evaluate = functools.partial(_evaluate_expression, modulus=self._modulus)
    labels = jax.vmap(evaluate)(expressions[:,:eff_length])
    labels = jnn.one_hot(labels, self._modulus)

    #one_hot_expressions = jnn.one_hot(expressions, self._modulus + len(self._operators))
    one_hot_expressions = self.fa_transform(rng_trans, expressions, sequence_length)
    '''
    one_hot_expressions, raw_x = self.fa_transform(rng_trans, batch_size, sequence_length)
    evaluate = functools.partial(_evaluate_expression, modulus=self._modulus)
    labels = jax.vmap(evaluate)(raw_x[:, :eff_length])
    labels = jnn.one_hot(labels, self._modulus)
    '''
    return {'input': one_hot_expressions, 'output': labels,}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self.fa_dim
    #return self._modulus + len(self._operators)

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._modulus

  def fa_transform(self, rng, x, sequence_length):
    # 8 --> 3
    '''
    0,2,4th --> 0~4 / 1,3,5...th --> 5~7
    '''

    #rng1, rng2, rng3, rng4, rng5, rng6, pos = jax.random.split(rng, 7)
    '''
    rand_add_2 = jrandom.randint(rng1, shape=remainders.shape, minval=0, maxval=2)
    rand_add_3 = jrandom.randint(rng2, shape=remainders.shape, minval=0, maxval=3)
    op_rand_add_2 = jrandom.randint(rng3, shape=operations.shape, minval=0, maxval=2)
    op_rand_add_3 = jrandom.randint(rng4, shape=operations.shape, minval=0, maxval=3)
    
    batch = jnp.empty((batch_size, sequence_length), dtype=int)
    batch = batch.at[:, ::2].set(jnp.where(remainders<4, remainders*2+rand_add_2, 3+remainders))
    x = batch.at[:, 1::2].set(jnp.where(operations<4, (operations-self._modulus)*3+op_rand_add_3, operations-1+op_rand_add_2))
    '''

    '''
    remainders = jax.random.randint(rng5,(batch_size, (sequence_length+1) // 2 ), 0,self._modulus)
    operations = jax.random.randint(rng6,(batch_size, (sequence_length) // 2 ), 0,3)
    rand_add_2 = jrandom.randint(rng1, (batch_size, (sequence_length+1) // 2 ), minval=0, maxval=2)
    rand_add_3 = jrandom.randint(rng2, (batch_size, (sequence_length+1) // 2 ), minval=0, maxval=3)
    op_rand_add_2 = jrandom.randint(rng3, (batch_size, (sequence_length) // 2 ), minval=0, maxval=2)
    op_rand_add_3 = jrandom.randint(rng4, (batch_size, (sequence_length) // 2 ), minval=0, maxval=3)

    batch = jnp.empty((batch_size, sequence_length), dtype=int)
    batch = batch.at[:, ::2].set(jnp.where(remainders<3, remainders*2+rand_add_2, 3+remainders))
    x = batch.at[:, 1::2].set(jnp.where(operations<2, operations*3+op_rand_add_3, 6+op_rand_add_2))
    '''

    #'''
    rng1, rng2, pos = jax.random.split(rng, 3)
    rand_add_2 = jrandom.randint(rng1, shape=x.shape, minval=0, maxval=2)
    rand_add_3 = jrandom.randint(rng2, shape=x.shape, minval=0, maxval=3)
    x = jnp.where(x < 5, jnp.where(x<3, x*2+rand_add_2, x+3), jnp.where(x < 7, x*3-15+rand_add_3, x - rand_add_2))
    #'''

    ''' It works
    batch = jnp.empty((batch_size, sequence_length), dtype=int)
    batch_raw_x = jnp.empty((batch_size, sequence_length), dtype=int)
    remainders = jrandom.choice(rng1, jax.numpy.array([0,1,2,3,4,5,6,7]),shape=(batch_size,(sequence_length+1)//2), p=jax.numpy.array([0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2]))
    operations = jrandom.choice(rng2, jax.numpy.array([0,1,2,3,4,5,6,7]),shape=(batch_size,sequence_length//2), p=jax.numpy.array([1/9,1/9,1/9,1/9,1/9,1/9,1/6,1/6]))
    batch = batch.at[:, ::2].set(remainders)
    x = batch.at[:, 1::2].set(operations)
    batch_raw_x = batch_raw_x.at[:, ::2].set(jnp.where(remainders<4, jnp.where(remainders<2,0,1), jnp.where(remainders<6,2,jnp.where(remainders<7,3,4))))
    raw_x = batch_raw_x.at[:, 1::2].set(jnp.where(operations<6,jnp.where(operations<3,5,6), 7))
    '''

    x_1 = x % 2
    x = x // 2
    x_2 = x % 2
    x = x // 2
    x_3 = x % 2
    if self.fa_dim == 4:
      stripe = jnp.zeros(shape=x.shape)
      if sequence_length > 1:
        position = jrandom.randint(pos, shape=([x.shape[0]]), minval=1, maxval=sequence_length)
        stripe = stripe.at[jnp.arange(x.shape[0]), position].set(1)
      x = jnp.stack((stripe, x_1, x_2, x_3), axis=2)
    else:
      x = jnp.stack((x_1, x_2, x_3), axis=2)
    return x