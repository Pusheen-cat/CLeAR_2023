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

"""Training loop for base generalization experiments."""

import dataclasses
import functools
import random
from typing import Tuple, List, Callable, Mapping, Optional, Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from tasks import task as task_lib
from training import curriculum as curriculum_lib
from training import Old_range_evaluation as range_evaluation

import copy

from jax.tree_util import tree_map, tree_reduce

##
from functools import partial
from jax import jit
##

_LossMetrics = Optional[Mapping[str, jnp.ndarray]]
_LossFn = Callable[[list, chex.Array, int], Tuple[float, _LossMetrics]] #! Callable[[chex.Array, chex.Array], Tuple[float, _LossMetrics]]
_AccuracyFn = Callable[[chex.Array, chex.Array], float]
#_ModelApplyFn = Callable[..., chex.Array]
_MAX_RNGS_RESERVE = 5_000_000


@dataclasses.dataclass
class ClassicTrainingParams:
  """Parameters needed to train classical architectures."""
  seed: int  # Used to sample during forward pass (e.g. from final logits).
  model_init_seed: int  # Used to initialize model parameters.
  training_steps: int
  log_frequency: int

  task: list #task_lib.GeneralizationTask
  tasks: list #!
  joint_training: bool
  length_curriculum: curriculum_lib.Curriculum
  batch_size: int

  model: hk.Transformed
  loss_fn: Callable[[list, jnp.ndarray, int], Tuple[float, _LossMetrics]] #!Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, _LossMetrics]]
  my_cl_loss_fn: Callable[[list, jnp.ndarray, int], Tuple[float, _LossMetrics]]
  learning_rate: float

  pretune: dict # @3/18
  debug: bool

  #! add 05/04
  cl_setting: tuple

  test_model: Optional[hk.Transformed] = None
  max_grad_norm: float = 1.0 #! defult: 1.
  is_autoregressive: bool = False

  compute_full_range_test: bool = False
  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100

  accuracy_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                 jnp.ndarray]] = None

  #! @add 3/22
  share_input_size: int = 0
  lr_schedule: bool = False




def _apply_loss_and_metrics_fn(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    model_apply_fn,#: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    task_id: int,
    init_para : bool,
    share_input_size : int,
    cl_setting:tuple,
    is_autoregressive: bool = False,
    prev_params = None,
    additional_ = None,
) -> Tuple[float, Tuple[_LossMetrics, float]]:
  """Computes the model output and applies the loss function.

  Depending on whether a model is autoregressive or not, it will have a
  different number of input parameters (i.e., autoregressive models also require
  the targets as an input).

  Args:
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The loss of the model for the batch of data, extra loss metrics and the
    accuracy, if accuracy_fn is not None.
  """
  if is_autoregressive:
      outputs = model_apply_fn(params, rng_key, batch["input"], batch["output"], sample=False)
  else:
      outputs = model_apply_fn(params, rng_key, batch["input"], task_id = task_id, init_para = init_para, share_input_size =share_input_size)

  #here
  if cl_setting[1] == 'lwf' and task_id>0:
      prev_outputs = model_apply_fn(prev_params, rng_key, jax.lax.stop_gradient(batch["input"]), task_id=task_id, init_para=init_para, share_input_size=share_input_size)
  else:
      prev_outputs = None

  if cl_setting[1] == 'ewc' and task_id > 0:
      prev_outputs = tree_reduce(lambda x, y: jnp.sum(x) + jnp.sum(y),tree_map(lambda a, b, w: jnp.sum(w * (a - b) ** 2.0), params, jax.lax.stop_gradient(prev_params), jax.lax.stop_gradient(additional_),),)

  loss, loss_metrics, debug = loss_fn(outputs, prev_outputs, batch["output"], task_id) #!

  if accuracy_fn is not None:
    accuracy = accuracy_fn(outputs[task_id], batch["output"], task_id) #! this is correct
  else:
    accuracy = None
  return loss, (loss_metrics, accuracy, debug)


@functools.partial(
    jax.jit,
    static_argnames=(
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
        "task_id",
        "init_para",
        "share_input_size",
        "is_autoregressive",
        "cl_setting",
        "get_grad",

    ),
)
def _update_parameters(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    model_apply_fn,#: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    task_id: int,
    init_para : bool,
    share_input_size : int,
    cl_setting: tuple,
    is_autoregressive: bool = False,
    prev_params = None,
    additional_ = None,
    get_grad = None,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  """Applies a single SGD update step to the model parameters.

  Args:
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    optimizer: The optimizer that computes the updates from the gradients of the
      `loss_fn` with respect to the `params` and the previous `opt_state`.
    opt_state: The optimizer state, e.g., momentum for each variable when using
      Adam.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The updated parameters, the new optimizer state, and the loss, loss metrics
    and accuracy.
  """
  (loss, (metrics, accuracy, debug)), grads = jax.value_and_grad(_apply_loss_and_metrics_fn, has_aux=True)(
      params, rng_key, batch, model_apply_fn, loss_fn, accuracy_fn, task_id, init_para, share_input_size, cl_setting, is_autoregressive, prev_params,additional_,
  )
  if get_grad:
      return grads
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  debug['lr'] = new_opt_state.hyperparams['learning_rate']
  return new_params, new_opt_state, (loss, metrics, accuracy, debug)


class TrainingWorker:
  """Training worker."""

  def __init__(self,
               training_params: ClassicTrainingParams,
               use_tqdm: bool = False):
    """Initializes the worker.
    Args:
      training_params: The training parameters.
      use_tqdm: Whether to add a progress bar to stdout.
    """
    self._training_params = training_params
    self._use_tqdm = use_tqdm

  def create_learning_rate_fn(self, base_learning_rate, warmup_step, total_step):
      """Creates learning rate schedule."""
      warmup_fn = optax.linear_schedule(
          init_value=0., end_value=base_learning_rate,
          transition_steps=warmup_step)
      constant = optax.linear_schedule(
          init_value=base_learning_rate, end_value=base_learning_rate,
          transition_steps=total_step-warmup_step)

      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, constant],
          boundaries=[warmup_step])
      return schedule_fn

  def run(
      self
  ) -> Tuple[List[Mapping[str, Any]], Optional[List[Mapping[str, Any]]],
             chex.ArrayTree]:
    """Trains the model with the provided config.

    Returns:
      Results (various training and validation metrics), module parameters
      and router parameters.
    """
    training_params = self._training_params
    rngs_reserve = min(_MAX_RNGS_RESERVE, 2*training_params.training_steps*len(training_params.tasks))

    random.seed(training_params.seed)
    np.random.seed(training_params.seed)
    rng_seq = hk.PRNGSequence(training_params.seed)
    rng_seq.reserve(rngs_reserve) #random seed

    #@ for pre_tune
    pre_tune_rng_seq = hk.PRNGSequence(training_params.seed)
    pre_tune_rng_seq.reserve(rngs_reserve)  # random seed

    results = []
    debug_result={}
    break_count = 0
    model = training_params.model
    task_list = training_params.task
    length_curriculum = training_params.length_curriculum

    @optax.inject_hyperparams
    def optimizer(learning_rate, eps=1e-8):
        return optax.chain(
            optax.clip_by_global_norm(training_params.max_grad_norm),
            optax.adam(learning_rate)
        )
    if training_params.lr_schedule:
        warmup_schedular = self.create_learning_rate_fn(training_params.learning_rate, 1000,
                                                        training_params.training_steps)
        optimizer = optimizer(warmup_schedular)
    else:
        optimizer = optimizer(training_params.learning_rate)


    dummy_batch_list = []
    for task in task_list:
        print(task)
        dummy_batch = task.sample_batch(
            next(rng_seq), length=10, batch_size=training_params.batch_size)
        print(dummy_batch['input'].shape)
        dummy_batch_list.append(dummy_batch)

    model_init_rng_key = jax.random.PRNGKey(training_params.model_init_seed)

    #! @ parameter initalization with running model with sample input
    if training_params.is_autoregressive:
      params = model.init(
          model_init_rng_key,
          dummy_batch_list[0]["input"],
          dummy_batch_list["output"],
          sample=False)
    else:
      params = model.init(model_init_rng_key, dummy_batch_list[0]["input"], share_input_size = training_params.share_input_size)

    if training_params.cl_setting[1] == 'lwf':
        self.lwf_initlal_parm = copy.deepcopy(params)

    print('>>> Model Parameters Keys <<<')
    print(params.keys())

    opt_state = optimizer.init(params)
    self._params, self._step = params, 0
    self.prev_params = None
    self.additional = None

    steps = range(training_params.training_steps + 1)
    for_task_eval_results = []
    num_task = len(training_params.task)
    for task_seq in range(num_task):
        if not training_params.joint_training:
            task_id = task_seq

        #! @Use pretune  #! made 3/18
        if training_params.pretune['tune'] and not training_params.joint_training and task_id>0:
            pre_steps = range(training_params.pretune['pretune_steps']+1)
            if self._use_tqdm:
                pre_steps = tqdm.tqdm(pre_steps, desc=f'Task {task_seq} pretune')
            for pre_step in pre_steps:
                length = length_curriculum.sample_sequence_length(pre_step)
                train_batch = task_list[task_id].sample_batch(
                    next(pre_tune_rng_seq), length=length, batch_size=training_params.batch_size)
                params_tmp, opt_state, (_, _, _, debug) = _update_parameters(
                    params=params,
                    rng_key=next(rng_seq),
                    batch=train_batch,
                    model_apply_fn=model.apply,
                    loss_fn=training_params.loss_fn,
                    accuracy_fn=training_params.accuracy_fn,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    task_id=task_id,
                    init_para=False,
                    share_input_size= training_params.share_input_size,
                    is_autoregressive=training_params.is_autoregressive,
                    cl_setting=(0,0)
                )
                for pretune_layer in training_params.pretune['train_layers']:
                    params[pretune_layer+f'{task_seq}'] = params_tmp[pretune_layer+f'{task_seq}']
            if self._use_tqdm:
                pre_steps.close()


        if self._use_tqdm:
            steps = tqdm.tqdm(steps, desc=f'Task {task_seq} TRAIN')

        #! @main training
        opt_state = optimizer.init(params)  #! new optimizer initialize for each task
        if training_params.cl_setting[1] == 'lwf':
            params = copy.deepcopy(self.lwf_initlal_parm)
        for step in steps:
            if training_params.joint_training:
                task_id = step%num_task
            # Randomness handled by either python.random or numpy.
            length = length_curriculum.sample_sequence_length(step)
            # Randomness handled by either jax, python.random or numpy.
            train_batch = task_list[task_id].sample_batch(
              next(rng_seq), length=length, batch_size=training_params.batch_size)
            params, opt_state, (train_loss, train_metrics, train_accuracy, debug) = _update_parameters(
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn if training_params.joint_training else training_params.my_cl_loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  task_id = task_id,
                  init_para = False,
                  share_input_size=training_params.share_input_size,
                  is_autoregressive=training_params.is_autoregressive,
                  prev_params = self.prev_params,
                  additional_ = self.additional, # importance weight
                  cl_setting = training_params.cl_setting
            )
            #! Here output become non-jit object
            self._params, self._step = params, step

            log_freq = training_params.log_frequency
            if (log_freq > 0) and (step % log_freq == 0):
                log_data = {
                    "step": step,
                    "train_loss": float(train_loss),
                }
            if training_params.accuracy_fn is not None:
              log_data["train_accuracy"] = float(train_accuracy)
            for key, value in train_metrics.items():
              log_data[".".join(["train_metrics", key])] = np.array(value)
            results.append(log_data)

            # We need to access this private attribute since the default reserve size
            # can not be edited yet.
            if not rng_seq._subkeys:  # pylint: disable=protected-access
                rng_seq.reserve(rngs_reserve)

            #@@@@@@@@@@@@@@@@@@@@@ Debug @@@@@@@@@@@@@@@@#
            if training_params.debug and step%100==0:
                if task_id>0:
                    print(debug['loss'], debug['reg'])
                # loss 0.7 -> E^-8
        if self._use_tqdm:
            steps.close()


        #@ Additional For CL Algorithm
        if training_params.cl_setting[1]=='ewc' and task_seq < num_task-1: # Line A (line A,B must be in order)
            fisher_steps = min(10_000, training_params.training_steps)
            fisher_steps_range = range(fisher_steps)
            fisher_diagonals = tree_map(lambda x: 0 * x, copy.deepcopy(params))
            num_samples = 0
            if self._use_tqdm:
                fisher_steps_range = tqdm.tqdm(fisher_steps_range, desc=f'Task {task_seq} Fisher for ewc')
            for step in fisher_steps_range:
                length = length_curriculum.sample_sequence_length(step)
                # Randomness handled by either jax, python.random or numpy.
                train_batch = task_list[task_id].sample_batch(
                    next(rng_seq), length=length, batch_size=training_params.batch_size)
                num_samples += training_params.batch_size
                grads = _update_parameters(
                    params=params,
                    rng_key=next(rng_seq),
                    batch=train_batch,
                    model_apply_fn=model.apply,
                    loss_fn=training_params.loss_fn if training_params.joint_training else training_params.my_cl_loss_fn,
                    accuracy_fn=training_params.accuracy_fn,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    task_id=task_id,
                    init_para=False,
                    share_input_size=training_params.share_input_size,
                    is_autoregressive=training_params.is_autoregressive,
                    prev_params=self.prev_params,
                    additional_=self.additional,  # importance weight
                    get_grad= True,
                    cl_setting=training_params.cl_setting
                )
                fisher_diagonals = tree_map(lambda a, b: a ** 2 + b, grads, fisher_diagonals)
            if self._use_tqdm:
                fisher_steps_range.close()
            self.additional = tree_map(lambda x: x / num_samples, fisher_diagonals)

        # save parameter to use next task !!!! must be in order !!!!
        if training_params.cl_setting[1]=='lwf' or training_params.cl_setting[1]=='ewc': #Line B
            self.prev_params = copy.deepcopy(params)

        if training_params.cl_setting[1]=='er':
            a=11




        eval_results = []
        for in_task_id, task in enumerate(task_list):
            if training_params.compute_full_range_test:
                eval_params = range_evaluation.EvaluationParams(
                    model=training_params.test_model or model,
                    params=params,
                    accuracy_fn=training_params.accuracy_fn,
                    sample_batch=task.sample_batch,
                    max_test_length=training_params.max_range_test_length,
                    total_batch_size=training_params.range_test_total_batch_size,
                    sub_batch_size=training_params.range_test_sub_batch_size,
                    tasks = training_params.tasks,
                    current_task = task_seq,
                    joint_training = training_params.joint_training,
                    task_id = in_task_id,
                    is_autoregressive=training_params.is_autoregressive,
                    share_input_size = training_params.share_input_size,
                )
                eval_result = range_evaluation.range_evaluation(
                  eval_params, use_tqdm=True)
            eval_results.append(eval_result)
        for_task_eval_results.append(eval_results)
    jax.clear_backends() #jax.clear_cachs()
    #jax.clear_caches()
    return results, for_task_eval_results, params, debug_result
