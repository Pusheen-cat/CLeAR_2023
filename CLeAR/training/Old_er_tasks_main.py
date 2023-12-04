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

"""Example script to train and evaluate a network."""

from absl import app

import haiku as hk
import jax.numpy as jnp
import numpy as np
import sys

from training import curriculum as curriculum_lib
from training import Old_er_tasks_training as training
from training import Old_tasks_utils as utils

import os
import time
import pickle as pkl

def main(unused_argv={'developing':True}) -> None:
    # Change your hyperparameters here. See constants.py for possible tasks and
    # architectures.
    print('>>>> unused_argv', unused_argv, file=sys.stdout)
    keys = unused_argv.keys()
    def check_dict(name, value):
        if name in keys:
            if value != unused_argv[name]:
                print(f'>> Changed {name} : {value} to {unused_argv[name]}')
            return unused_argv[name]
        else:
            if value == 777:
                print(f'Value for {name} must be determined')
                raise
            unused_argv[name] = value
            return value
    cl_setting = check_dict('cl_setting', (0, 0, 0))
    joint_training = check_dict('joint_training', False)
    assert (joint_training and cl_setting[0]) == False, '<Problem Setting Err> CL and Joint training can not done together'

    add_name = check_dict('add_name','')
    par2 = check_dict('par2',0)
    debug_arg = check_dict('debug',False)

    batch_size = check_dict('batch_size', 128)
    sequence_length = check_dict('sequence_length', 40)
    tasks = check_dict('tasks',['parity_check', 'modular_arithmetic', 'even_pairs', 'cycle_navigation'])  # ! run in this order
    architecture = check_dict('architecture', 'tape_rnn')
    architecture_params = check_dict('architecture_params',{'hidden_size': 256, 'memory_cell_size': 8, 'memory_size': 40})

    #@@ 3/18 - apply pretune task-wise layer before trainging whole network
    pretune = check_dict('pretune', {'tune': False, 'pretune_steps': 1_000, 'train_layers': ['lin_in_','lin_out_'],})
    '''
    'lin_in_' refers 'lin_in_{task_idx}' ;; 'lin_in_0', 'lin_in_1', 'lin_in_2', 'lin_in_3'
    '''
    projection_dim = check_dict('projection_dim', False)

    # 5/18 univ_setting
    univ_setting = check_dict('univ_setting',0)
    fa_dim = check_dict('fa_dim', 3)

    #5/19 old_setting
    share_input = True
    use_raw = check_dict('use_raw',777)
    simple_share = check_dict('simple_share', False)
    if use_raw and not simple_share:
        share_input = False
        if projection_dim == False:
            print('use_raw, not simple share but No projection dim given')
            raise
    if not use_raw or simple_share:
        pretune['train_layers'] = ['lin_out_']
        projection_dim = False


    if use_raw:
        from training import Old_constants_raw as constants
    else:
        from training import Old_constants_mapped as constants



    # Create the task.
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, sequence_length + 1)))

    # Create the model.
    is_autoregressive = check_dict('is_autoregressive', False)
    computation_steps_mult = check_dict('computation_steps_mult', 0)

    task_list = []
    output_size_list = []
    #input_size_list = []  #no need for CFA that share input space
    pure_input_size_list =[]
    for _idx, each_task in enumerate(tasks): #ex. task: ['parity_check', 'modular_arithmetic', 'even_pairs', 'cycle_navigation']
        if univ_setting == 0:
            if use_raw:
                task = constants.TASK_BUILDERS[each_task]()  # input data dim to 3
            else:
                task = constants.TASK_BUILDERS[each_task](fa_dim)
        elif univ_setting == 1:
            # Used for High-correlation CL task of CN, MA etc.. (Table 2 of paper)
            modulus = check_dict('complexity_parameter_hc', [2, 3, 4, 5, 6, 7, 8])[_idx]
            if use_raw:
                task = constants.TASK_BUILDERS[each_task](modulus)
            else:
                task = constants.TASK_BUILDERS[each_task](fa_dim, modulus)
        task_list.append(task)
        output_size = task.output_size
        output_size_list.append(output_size)
        input_size = task.input_size
        #input_size_list.append(input_size + 1 + computation_steps_mult)  #no need for CFA that share input space
        pure_input_size_list.append(input_size)
    share_input_dim = max(pure_input_size_list) + 1 + int(computation_steps_mult>0)
    each_input_dim_list = pure_input_size_list+ [1 + int(computation_steps_mult>0) for i in range(len(pure_input_size_list))]


    # Create the model.
    single_output_list = [task.output_length(10) == 1 for task in task_list]
    single_output = True  #! temporary
    model = constants.MODEL_BUILDERS[architecture]( #! --> CL_rnn make_rnn
        output_size=output_size_list,
        #input_size=input_size_list, #no need for CFA that share input space
        return_all_outputs=True,
        num_task=len(tasks),
        projection_dim = projection_dim,
        each_input_dim_list = each_input_dim_list,
        **architecture_params)
    if is_autoregressive:
        if 'transformer' not in architecture:
            model = utils.make_model_with_targets_as_input(
                model, computation_steps_mult)

        model = utils.add_sampling_to_autoregressive_model(model, single_output_list)
    else: #! Default
        model = utils.make_model_with_empty_targets(
            model, task_list, computation_steps_mult, single_output_list, share_input, each_input_dim_list
        )
    model = hk.transform(model)


    # Create the loss and accuracy based on the pointwise ones.
    def loss_fn(output, prev_outputs, target, idx): #! for pretune
        #print('loss_fn output',output.shape) #(128, 5)
        #print('loss_fn target', target.shape) #(128, 5)
        loss = task.CFA_loss_fn(output, target, idx)
        return loss, {}, {}

    def my_cl_loss_fn(output, additional, target, idx, par2=par2):
        #print('my_cl_loss_fn output', len(output), output[0].shape, output[1].shape) #3, (128, 5)
        #print('my_cl_loss_fn target', target.shape) #(128, 5)
        if cl_setting[1] == 'er':
            loss = task.CFA_loss_fn(output, target, idx)
            debug = {}
        elif cl_setting[1] == 'lwf': # CL using previous task model
            loss, debug = task.Old_lwf_loss_fn(output, additional, target, idx, cl_setting[2])
        elif cl_setting[1] == 'ewc':
            loss, debug = task.Old_ewc_loss_fn(output, additional, target, idx, cl_setting[2])

        else: # CL without using freezed previous task model
            loss = task.CFA_loss_fn(output, target, idx)
            debug={}
        return loss, {}, debug

    def accuracy_fn(output, target, idx):
        #print('accuracy_fn output',output.shape) #(128, 5)
        #print('accuracy_fn target', target.shape) #(128, 5)
        mask = task_list[idx].accuracy_mask(target)
        return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

    # Create the final training parameters.
    training_params = training.ClassicTrainingParams(
        seed=check_dict('seed', 0),
        model_init_seed=check_dict('model_init_seed', 0),
        training_steps=check_dict('training_steps', 50_000),  # ! 10_000
        log_frequency=check_dict('log_frequency', 100),
        length_curriculum=curriculum,
        batch_size=batch_size,
        task=task_list,  # !
        tasks=tasks,  # !
        joint_training=joint_training,  # !
        model=model,
        loss_fn=loss_fn,
        my_cl_loss_fn = my_cl_loss_fn,
        learning_rate=check_dict('learning_rate', 1e-3),
        accuracy_fn=accuracy_fn,
        compute_full_range_test=check_dict('compute_full_range_test', True),
        max_range_test_length=check_dict('max_range_test_length', 100),
        range_test_total_batch_size=check_dict('range_test_total_batch_size', 512),
        range_test_sub_batch_size=check_dict('range_test_sub_batch_size', 64),
        is_autoregressive=is_autoregressive,
        debug = debug_arg,
        pretune=pretune,
        share_input_size = share_input_dim if share_input else 0,
        lr_schedule = check_dict('lr_schedule', False),
        cl_setting = cl_setting,
        train_sequence_max = sequence_length,

    )

    training_worker = training.TrainingWorker(training_params, use_tqdm=True)
    _, eval_results, _, debug_result = training_worker.run()

    # Here, insert code for saving results

if __name__ == '__main__':
    '''
    dict_ = {}
    dict_['architecture'] = 'tape_rnn'
    main(dict_)
    '''
    unused_argv = {'developing': True, 'share_input': True, 'joint_training': False, 'architecture': 'lstm',
                   'architecture_params': {'hidden_size': 256, },
                   'training_steps': 10_000, 'tasks': ['dual_ep_pc', 'dual_ep_pc'], 'par2':3, 'debug': False}
    main(unused_argv)
    #!app.run(main)