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

"""Builders for RNN/LSTM cores."""

from typing import Callable, Any, Type

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from functools import partial
from jax import jit

from models import tape_rnn


def make_rnn(
        output_size: list, #! output_size_list := [task.output_size for task]
        rnn_core: Type[hk.RNNCore],
        return_all_outputs: bool = False, #! True fo default
        input_window: int = 1,
        num_task=1, #! len(tasks)
        projection_dim=0, #! share_input_dim if share_input else 0,
        each_input_dim_list = None,
        **rnn_kwargs: Any) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Returns an RNN model, not haiku transformed.

    Only the last output in the sequence is returned. A linear layer is added to
    match the required output_size.

    Args:
      output_size: The list of output size of the model.
      rnn_core: The haiku RNN core to use. LSTM by default.
      return_all_outputs: Whether to return the whole sequence of outputs of the
        RNN, or just the last one.
      input_window: The number of tokens that are fed at once to the RNN.
      **rnn_kwargs: Kwargs to be passed to the RNN core.
    """

    @partial(jit, static_argnums=1)
    def func(x, axis):
        return x.min(axis)

    def rnn_model(x: jnp.array, input_length: int = 1 , task_id: int=0, init_para: bool=False) -> jnp.ndarray:
        #!print('x',x.shape) #(128, input length +1 , 4) for task cyclic navigation / parity check.. etc (after mapping)


        core = rnn_core(**rnn_kwargs)
        if issubclass(rnn_core, tape_rnn.TapeRNNCore):
            initial_state = core.initial_state(x.shape[0], input_length)  # pytype: disable=wrong-arg-count
        else:
            initial_state = core.initial_state(x.shape[0])

        out_linears = {}
        if projection_dim:

            in_linears = {}
            for task_head in range(num_task):
                in_linears[task_head] = hk.Linear(projection_dim, name=f'lin_in_{task_head}')
        for task_head in range(num_task):
            out_linears[task_head] = hk.Linear(output_size[task_head], name=f'lin_out_{task_head}')

        if projection_dim:
            ###
            #'''
            batch_size, seq_length,_ = x.shape
            mul = jnp.zeros([num_task])
            mul = mul.at[task_id].set(1)
            tot_input = jnp.zeros([batch_size, seq_length, sum(each_input_dim_list)])
            tot_input = tot_input.at[:, :, sum(each_input_dim_list[:task_id]):sum(each_input_dim_list[:task_id + 1])].set(x)
            x = in_linears[0](tot_input[:, :, 0:each_input_dim_list[0]]) * mul[0]
            for task_head in range(1, num_task):
                x += in_linears[task_head](
                    tot_input[:, :, sum(each_input_dim_list[:task_head]):sum(each_input_dim_list[:task_head + 1])]) * mul[
                          task_head]
            ###
            #'''
            #x = in_linears[task_id](x)
            x = jnn.relu(x)


        batch_size, seq_length, embed_size = x.shape
        if seq_length % input_window != 0:
            x = jnp.pad(x, ((0, 0), (0, input_window - seq_length % input_window),
                            (0, 0)))
        new_seq_length = x.shape[1]
        x = jnp.reshape(
            x,
            (batch_size, new_seq_length // input_window, input_window, embed_size))
        # ! print('1',x.shape) # (128, 26:input seq length, 1, 3)
        x = hk.Flatten(preserve_dims=2)(x)
        # ! print('2',x.shape) # (128, 26:input seq length, 3)

        output, _ = hk.dynamic_unroll(
            core, x, initial_state, time_major=False, return_all_states=True)
        output = jnp.reshape(output, (batch_size, new_seq_length, output.shape[-1]))

        if not return_all_outputs:
            output = output[:, -1, :]  # (batch, time, alphabet_dim)
        output = jnn.relu(output)

        output_ = []
        for task_head in range(num_task):
            output_.append(out_linears[task_head](output))

        return output_

    # ! keep in mind that output is post processed with "output = output[:, -output_length:]" in CL_utils line 93
    # ! and loss is calculate with loss_fn in CL_main line 105

    return rnn_model
