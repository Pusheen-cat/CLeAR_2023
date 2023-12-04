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

"""Base class for generalization tasks."""

import abc
from typing import TypedDict

import chex
import jax #!
import jax.nn as jnn
import jax.numpy as jnp

Batch = TypedDict('Batch', {'input': chex.Array, 'output': chex.Array})


class GeneralizationTask(abc.ABC):
  """A task for the generalization project.

  Exposes a sample_batch method, and some details about input/output sizes,
  losses and accuracies.
  """

  @abc.abstractmethod
  def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
                   length: int) -> Batch:
    """Returns a batch of inputs/outputs."""

  def pointwise_loss_fn(self, output: chex.Array,
                        target: chex.Array) -> chex.Array:
    """Returns the pointwise loss between an output and a target."""
    #! returns: (128, 5)
    return -target * jnn.log_softmax(output) #! loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))

  def accuracy_fn(self, output: chex.Array, target: chex.Array) -> chex.Array:
    """Returns the accuracy between an output and a target."""
    return (jnp.argmax(output,axis=-1) == jnp.argmax(target,axis=-1)).astype(jnp.float32)

  def accuracy_mask(self, target: chex.Array) -> chex.Array:
    """Returns a mask to compute the accuracies, to remove the superfluous ones."""
    # Target is a shape of shape (B, T, C) where C is the number of classes.
    # We want a mask per input (B, T), so we take this shape.
    return jnp.ones(target.shape[:-1])

  @property
  @abc.abstractmethod
  def input_size(self) -> int:
    """Returns the size of the input of the models trained on this task."""

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    """Returns the size of the output of the models trained on this task."""

  def output_length(self, input_length: int) -> int:
    """Returns the length of the output, given an input length."""
    del input_length
    return 1

#----------------------------------------------------------------------------------------------------------------------#
  #CFA
  def CFA_loss_fn(self, output: list, target: chex.Array, par1) -> chex.Array:
    """
    This loss is for cl_setting[1] : False and Pre-tune
    Returns the pointwise loss between an output and a target.
    """
    #! This is for pre-tune phase
    return jnp.mean(jnp.sum(-target * jnn.log_softmax(output[par1]), axis=-1))
  def CFA_premodel_loss_fn(self, output: list, pre_output: list, target: chex.Array, par1, cl_setting_2) -> chex.Array:
    """
    This loss is for cl_setting[1] : True
    Returns the pointwise loss between an output and a target.
    print(output[0]) # float32[128,2]
    print(target) # float32[128,2]
    """
    #! This is for pre-tune phase
    loss = jnp.mean(jnp.sum(-target * jnn.log_softmax(output[par1]), axis=-1))
    for idx in range(par1):
      if cl_setting_2 == 0:
        loss += jnp.mean(jnp.sum(-jax.lax.stop_gradient(pre_output[idx]) * jnn.log_softmax(output[idx]), axis=-1))
      elif cl_setting_2 == 1:
        loss += jnp.mean(jnp.sum(-jnn.one_hot(jnp.argmax(jax.lax.stop_gradient(pre_output[idx]),axis=-1), num_classes=output[idx].shape[-1]) * jnn.log_softmax(output[idx]), axis=-1))
    return loss, {}
  def CFA_noise_loss_fn(self, output: list, pre_output: list, target: chex.Array, par1, cl_setting_2) -> chex.Array:
    """
    This loss is for cl_setting[1] : True
    Returns the pointwise loss between an output and a target.
    print(output[0]) # float32[128,2]
    print(target) # float32[128,2]
    """
    #! This is for pre-tune phase
    if par1 == 0:
      loss = jnp.mean(jnp.sum(-target * jnn.log_softmax(output[par1]), axis=-1))
    else:
      loss = jnp.mean(jnp.sum(-jnn.one_hot(jnp.argmax(jax.lax.stop_gradient(pre_output[0]),axis=-1), num_classes=output[0].shape[-1]) * jnn.log_softmax(output[0]), axis=-1))
    return loss, {}

  def debug_CFA_premodel_loss_fn(self, output: list, pre_output: list, target: chex.Array, par1, cl_setting_2) -> chex.Array:
    """
    This loss is for cl_setting[1] : True
    Returns the pointwise loss between an output and a target.
    print(output[0]) # float32[128,2]
    print(target) # float32[128,2]
    """
    loss_list = []
    #! This is for pre-tune phase
    loss = jnp.mean(jnp.sum(-target * jnn.log_softmax(output[par1]), axis=-1))
    loss_list.append(loss)
    for idx in range(par1):
      if cl_setting_2 == 0:
        tmp = jnp.mean(jnp.sum(-jax.lax.stop_gradient(pre_output[idx]) * jnn.log_softmax(output[idx]), axis=-1))
        loss += tmp
        loss_list.append(tmp)
      elif cl_setting_2 == 1:
        tmp = jnp.mean(jnp.sum(-jnn.one_hot(jnp.argmax(jax.lax.stop_gradient(pre_output[idx]),axis=-1), num_classes=output[idx].shape[-1]) * jnn.log_softmax(output[idx]), axis=-1))
        loss += tmp
        loss_list.append(tmp)
    return loss, {'loss_list':loss_list}

  #!@@@ here is loss that I modified to make CL
  def my_pointwise_loss_fn(self, output: list, target: chex.Array, par1) -> chex.Array:
    """Returns the pointwise loss between an output and a target."""
    #! This is for pre-tune phase
    return jnp.sum(-target[par1] * jnn.log_softmax(output[par1]), axis=-1)

  def my_cl_loss_fn(self, output: list, target: chex.Array, par1:int, par2: int) -> chex.Array:
    """Returns the pointwise loss betweoutputen an output and a target."""
    #! print(output[0]) # float32[128,2]
    #! print(output[1]) # float32[128,5]
    #! print(output[2]) # float32[128,2]
    #! print(target) # float32[128,2]
    #! print(-target * jnn.log_softmax(output[par1])) # float32[128,2]
    #!jax.debug.print("output{output}", output=target)
    loss = jnp.sum(-target[par1] * jnn.log_softmax(output[par1]), axis=-1)
    #'''
    for idx in range(par1):
      prediction = jnn.softmax(output[idx])
      loss_ = jnp.sum(-target[idx] * jnn.log_softmax(output[idx]), axis=-1)
      if par2 ==0: #! stable w/ stack rnn, lstm | Fail w/ rnn, tape_rnn  | p lstm rnn tape_rnn/ n
        pseudo_target = jnn.softmax(output[idx])
        loss_pre = jnp.sum(-pseudo_target * pseudo_target, axis = -1)
      elif par2 ==1: #! stable w/ rnn, stack rnn, lstm | Fail w/ tape_rnn  | p rnn/ n lstm tape_rnn
        pseudo_target = jax.lax.stop_gradient(jnn.hard_sigmoid(6 * (jnn.softmax(output[idx]) - 0.5) / 0.7))  # 200 #! o -rnn/lstm/
        loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
      elif par2 == 2: #! stable w/ stack rnn, lstm | Fail w/ rnn, tape_rnn  | p rnn/ n lstm tape_rnn
        pseudo_target = jax.lax.stop_gradient(jnn.one_hot(jnp.argmax(output[idx], axis=-1), 2))
        loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
      elif par2 == 3: #! stable w/ stack rnn, lstm / Fail w/ rnn, tape_rnn  | p rnn/ n lstm tape_rnn
        pseudo_target = jnn.softmax(output[idx], axis=-1)
        loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
      elif par2 == -1: #! stable w/ tape, stack, naive rnn, lstm
        loss_pre = jnp.sum(-target[idx] * jnn.log_softmax(output[idx]), axis=-1)
      loss = loss_pre

    #'''
    loss = jnp.mean(loss)

    return loss

  def my_debug_loss_fn(self, output: list, target: chex.Array, par1:int, par2: int,) -> chex.Array:
    """Returns the pointwise loss betweoutputen an output and a target."""
    debug = {}
    #! print(output[0]) # float32[128,2]
    #! print(output[1]) # float32[128,5]
    #! print(output[2]) # float32[128,2]
    #! print(target) # float32[128,2]
    #! print(-target * jnn.log_softmax(output[par1])) # float32[128,2]

    #!jax.debug.print("output{output}", output=target)
    loss = jnp.sum(-target[par1] * jnn.log_softmax(output[par1]), axis=-1)
    prediction = jnn.softmax(output[par1])
    debug['loss_true_label'] = jnp.mean(loss)
    debug['prediction'] = prediction
    debug['label'] = target[par1]
    debug['incorrect_count'] = jnp.sum(jnp.abs(jnp.argmax(prediction, axis=-1)-jnp.argmax(target[par1], axis=-1)))
    #'''
    for idx in range(par1):
      prediction = jnn.softmax(output[idx])
      loss_ = jnp.sum(-target[idx] * jnn.log_softmax(output[idx]), axis=-1)
      if par2 ==0: #! stable w/ stack rnn, lstm / Fail w/ rnn, tape_rnn  | p lstm rnn tape_rnn/ n
        pseudo_target = jnn.softmax(output[idx])
        loss_pre = jnp.sum(-pseudo_target * pseudo_target, axis = -1)
      elif par2 ==1: #! stable w/ rnn, stack rnn, lstm / Fail w/ tape_rnn  | p rnn/ n lstm tape_rnn
        pseudo_target = jax.lax.stop_gradient(jnn.hard_sigmoid(6 * (jnn.softmax(output[idx]) - 0.5) / 0.7))  # 200 #! o -rnn/lstm/
        loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
      elif par2 == 2: #! stable w/ stack rnn, lstm / Fail w/ rnn, tape_rnn  | p rnn/ n lstm tape_rnn
        pseudo_target = jax.lax.stop_gradient(jnn.one_hot(jnp.argmax(output[idx], axis=-1), 2))
        loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
      elif par2 == 3: #! stable w/ stack rnn, lstm / Fail w/ rnn, tape_rnn  | p rnn/ n lstm tape_rnn
        pseudo_target = jnn.softmax(output[idx], axis=-1)
        loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
      elif par2 == -1: #! stable w/ tape, stack, naive rnn, lstm
        loss_pre = jnp.sum(-target[idx] * jnn.log_softmax(output[idx]), axis=-1)
      #'''
      err = jnp.sum(jnp.mean(jnp.abs(prediction-target[idx]), axis=0))
      #jax.debug.print("err {arg1} loss_pre {arg2} loss {arg3}", arg1=jnp.log(err), arg2 = jnp.log(jnp.mean(loss_pre.copy())), arg3 = jnp.log(jnp.mean(loss.copy())))
      debug['loss_true_label']=jnp.mean(loss_)
      debug['prediction']=prediction
      debug['label']=target[idx]
      debug['err']=err
      debug['loss_ssl'] = loss_pre
      debug['incorrect_count'] = jnp.sum(jnp.abs(jnp.argmax(prediction, axis=-1) - jnp.argmax(target[idx], axis=-1)))
      #'''

      loss = loss_pre

    #'''
    loss = jnp.mean(loss)

    return loss, debug
  '''
  w/o grad clipping (grad clip = 1.0)
  rnn all stable
  lstm par2=1,2 &step=50k fail, 
  
  w/ grad clipping to 0.1
  rnn 10000steps fail
  lstm 10000steps fail
  stack_rnn 50000steps fail
  
  w/ entropy reducing loss (par2==3)
  tape_rnn 10k steps fails with seed 0
  lstm 10k steps fails with seed 1
  tape_rnn 10k / lstm 50k fails with seed 2
  '''

  def my_debug_loss_fn_v2(self, output: list, target: chex.Array, par1:int, par2: int) -> chex.Array: #4/26
    """Returns the pointwise loss betweoutputen an output and a target."""
    debug = {}
    #! print(output[0]) # float32[128,2]
    #! print(output[1]) # float32[128,5]
    #! print(output[2]) # float32[128,2]
    #! print(target) # float32[128,2]
    #! print(-target * jnn.log_softmax(output[par1])) # float32[128,2]

    #!jax.debug.print("output{output}", output=target)
    idx=0
    prediction = jnn.softmax(output[idx])
    loss_ = jnp.sum(-target[idx] * jnn.log_softmax(output[idx]), axis=-1)
    if par2 ==0: #! stable w/ stack rnn, lstm / Fail w/ rnn, tape_rnn  | p lstm rnn tape_rnn/ n
      pseudo_target = jnn.softmax(output[idx])
      loss_pre = jnp.sum(-pseudo_target * pseudo_target, axis = -1)
    elif par2 ==1: #! stable w/ rnn, stack rnn, lstm / Fail w/ tape_rnn  | p rnn/ n lstm tape_rnn
      pseudo_target = jax.lax.stop_gradient(jnn.hard_sigmoid(6 * (jnn.softmax(output[idx]) - 0.5) / 0.7))  # 200 #! o -rnn/lstm/
      loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
    elif par2 == 2: #! stable w/ stack rnn, lstm / Fail w/ rnn, tape_rnn  | p rnn/ n lstm tape_rnn
      pseudo_target = jax.lax.stop_gradient(jnn.one_hot(jnp.argmax(output[idx], axis=-1), 2))
      loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
    elif par2 == 3: #! stable w/ stack rnn, lstm / Fail w/ rnn, tape_rnn  | p rnn/ n lstm tape_rnn
      pseudo_target = jnn.softmax(output[idx], axis=-1)
      loss_pre = jnp.sum(-pseudo_target * jnn.log_softmax(output[idx]), axis=-1)
    elif par2 == -1: #! stable w/ tape, stack, naive rnn, lstm
      loss_pre = jnp.sum(-target[idx] * jnn.log_softmax(output[idx]), axis=-1)
    #'''
    err = jnp.sum(jnp.mean(jnp.abs(prediction-target[idx]), axis=0))
    #jax.debug.print("err {arg1} loss_pre {arg2} loss {arg3}", arg1=jnp.log(err), arg2 = jnp.log(jnp.mean(loss_pre.copy())), arg3 = jnp.log(jnp.mean(loss.copy())))
    debug['loss_true_label']=jnp.mean(loss_)
    debug['prediction']=prediction
    debug['label']=target[idx]
    debug['err']=err
    debug['loss_ssl'] = loss_pre
    debug['incorrect_count'] = jnp.sum(jnp.abs(jnp.argmax(prediction, axis=-1) - jnp.argmax(target[idx], axis=-1)))
    #'''

    loss = loss_pre

    #'''
    loss = jnp.mean(loss)

    return loss, debug


  def Old_lwf_loss_fn(self, output: list, pre_output: list, target: chex.Array, par1, cl_setting_2) -> chex.Array:
    """
    This loss is for cl_setting[1] : True
    Returns the pointwise loss between an output and a target.
    print(output[0]) # float32[128,2]
    print(target) # float32[128,2]
    """
    #! This is for pre-tune phase
    loss = jnp.mean(jnp.sum(-target * jnn.log_softmax(output[par1]), axis=-1))
    for idx in range(par1):
      if cl_setting_2 == 0:
        loss += jnp.mean(jnp.sum(-jax.lax.stop_gradient(jnn.softmax(pre_output[idx])) * jnn.log_softmax(output[idx]), axis=-1))/par1
      elif cl_setting_2 == 1:
        loss += jnp.mean(jnp.sum(-jnn.one_hot(jnp.argmax(jax.lax.stop_gradient(pre_output[idx]),axis=-1), num_classes=output[idx].shape[-1]) * jnn.log_softmax(output[idx]), axis=-1))/par1
    return loss, {}

  def Old_ewc_loss_fn(self, output: list, additional: float, target: chex.Array, idx, cl_setting_2) -> chex.Array:
    """
    This loss is for cl_setting[1] : True
    Returns the pointwise loss between an output and a target.
    print(output[0]) # float32[128,2]
    print(target) # float32[128,2]
    """
    if cl_setting_2 < 0.001:
      print('lambda for ewc is near zero')
      raise
    #! This is for pre-tune phase
    loss = jnp.mean(jnp.sum(-target * jnn.log_softmax(output[idx]), axis=-1))
    loss_pre = None
    loss_ = None
    if idx>0:
      loss_pre=loss
      loss += additional*cl_setting_2
      loss_=additional*cl_setting_2
    return loss, {'loss': loss_pre, 'reg': loss_}