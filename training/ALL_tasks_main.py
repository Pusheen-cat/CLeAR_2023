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
import copy

from training import ALL_constants as constants
from training import curriculum as curriculum_lib
from training import ALL_tasks_training as training
from training import ALL_tasks_utils as utils

import os
import time
import pickle as pkl
import ast

def main(arg_dict={'developing':True}, neural_networks_chomsky_hierarchy=None) -> None:
    # Change your hyperparameters here. See constants.py for possible tasks and
    # architectures.
    unused_argv = copy.deepcopy(arg_dict)
    print('>>>> unused_argv', unused_argv, file=sys.stdout)
    keys = unused_argv.keys()
    def check_dict(name, value):
        if name in keys:
            if value != unused_argv[name]:
                print(f'>> Changed {name} : {value} to {unused_argv[name]}')
            return unused_argv[name]
        else:
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
    #@@ 3/22 - apply shared input to model
    share_input = check_dict('share_input', False)
    if share_input:
        pretune['train_layers'] = ['lin_out_']

    # 5/18 univ_setting
    univ_setting = check_dict('univ_setting',0)
    fa_dim = check_dict('fa_dim', 3)



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
        #Here
        if univ_setting == 0:
            #Use task with simple fa_dim as arg
            task = constants.TASK_BUILDERS[each_task](fa_dim)  # input data dim to 3
        elif univ_setting == 1:
            #Used for High-correlation CL task of CN, MA etc.. (Table 2 of paper)
            modulus = check_dict('complexity_parameter_hc', [2,3,4,5,6,7,8])[_idx]
            task = constants.TASK_BUILDERS[each_task](fa_dim, modulus)
        elif univ_setting == 2:
            #Used for Noise distribution experiment
            if each_task == 'noise_distribution': # Setting for task 'noise_distribution'
                task = constants.TASK_BUILDERS[each_task](fa_dim, check_dict('noise_idx', 0) + _idx,
                                                          check_dict('noise_var', 0.01))  # input data dim to 3
            else:
                task = constants.TASK_BUILDERS[each_task](fa_dim)  # input data dim to 3

        task_list.append(task)
        output_size = task.output_size
        output_size_list.append(output_size)
        input_size = task.input_size
        #input_size_list.append(input_size + 1 + computation_steps_mult)  #no need for CFA that share input space
        pure_input_size_list.append(input_size)
    share_input_dim = max(pure_input_size_list) + 1 + int(computation_steps_mult>0)


    # Create the model.
    single_output_list = [task.output_length(10) == 1 for task in task_list]
    single_output = True  #! temporary
    model = constants.MODEL_BUILDERS[architecture]( #! --> CL_rnn make_rnn
        output_size=output_size_list,
        #input_size=input_size_list, #no need for CFA that share input space
        return_all_outputs=True,
        num_task=len(tasks),
        share_input_dim = share_input_dim if share_input else 0,
        **architecture_params)
    if is_autoregressive:
        if 'transformer' not in architecture:
            model = utils.make_model_with_targets_as_input(
                model, computation_steps_mult)

        model = utils.add_sampling_to_autoregressive_model(model, single_output_list)
    else: #! Default
        model = utils.make_model_with_empty_targets(
            model, task_list, computation_steps_mult, single_output_list
        )
    model = hk.transform(model)


    # Create the loss and accuracy based on the pointwise ones.
    def loss_fn(output, prev_outputs, target, idx): #! for pretune
        #print('loss_fn output',output.shape) #(128, 5)
        #print('loss_fn target', target.shape) #(128, 5)
        loss = task.CFA_loss_fn(output, target, idx)
        return loss, {}, {}

    def my_cl_loss_fn(output, pre_output, target, idx, par2=par2):
        #print('my_cl_loss_fn output', len(output), output[0].shape, output[1].shape) #3, (128, 5)
        #print('my_cl_loss_fn target', target.shape) #(128, 5)
        if cl_setting[1]: # CL using previous task model
            if univ_setting <2:
                if debug_arg:
                    loss, debug = task.debug_CFA_premodel_loss_fn(output, pre_output, target, idx, cl_setting[2])
                else:
                    loss, debug = task.CFA_premodel_loss_fn(output, pre_output, target, idx, cl_setting[2])
            elif univ_setting ==2:
                loss, debug = task.CFA_noise_loss_fn(output, pre_output, target, idx, cl_setting[2])
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
        range_test_sub_batch_size=check_dict('range_test_sub_batch_size', 512),
        is_autoregressive=is_autoregressive,
        debug = debug_arg,
        pretune=pretune,
        share_input_size = share_input_dim if share_input else 0,
        lr_schedule = check_dict('lr_schedule', False),
        cl_setting = cl_setting

    )

    training_worker = training.TrainingWorker(training_params, use_tqdm=True)
    _, eval_results, _, debug_result = training_worker.run()

    #! "From here to end" : save result to log
    path_logs = '../logs/'

    #To Save loss
    if debug_arg:
        log_for_loss = path_logs+f'post_neurips/single'+str(tasks)+'-'+str(add_name)+'-'+str(architecture)
        loss_txt = open(log_for_loss+'.txt', 'w')
        for _key in debug_result.keys():
            loss_txt.write(_key+',')
            for loss in debug_result[_key]:
                loss_txt.write(str(loss.item())+',')
            loss_txt.write('\n')
        loss_txt.close()


    if check_dict('developing', False):
        path_logs += 'developing/'
    level_name_list = []
    for name in tasks:
        level = constants.TASK_LEVELS[name]
        temp = ''
        for tt in name.split('_'):
            temp += tt[0]
        level_name_list.append(level.upper()[0] + temp)
    not_sorted = level_name_list.copy()
    level_name_list.sort()
    for idx, name in enumerate(level_name_list):
        if not idx == 0:
            path_logs += "_"
        path_logs += name

    inner = ''
    for idx, name in enumerate(not_sorted):
        if not idx == 0:
            inner += "_"
        inner += name

    path_logs += '/'
    pkl_log = path_logs
    exel_log =pkl_log+ 'summary.xlsx'
    pkl_log += 'summary_05.pkl'
    exel_data = copy.deepcopy(unused_argv)
    path_logs += architecture + '-'
    if joint_training:
        path_logs += '[Joint_train]'
    else:
        path_logs += '[' + inner + ']'

    if not add_name == '':
        path_logs += '-' + add_name

    timestr = time.strftime("%Y%m%d-%H%M%S")  # 20120515-155045
    path_logs += '-' + timestr[2:8]  # YYMMDD
    exel_data['data'] = timestr[2:8]
    os.makedirs(path_logs, exist_ok=True)

    # Gather results and print final score. + write log
    f_sum = open(path_logs + '/summary.txt', 'w')
    f_sum.write(str(unused_argv) + '\n')
    f_sum.write("###\n")
    for task_id, train_task in enumerate(eval_results):
        f_sum.write(f'Trained: {train_task[0][0]["trained"]}\n')
        print(f'Trained: {train_task[0][0]["trained"]}')
        with open(path_logs + f'/{train_task[0][0]["trained"]}.txt', "w") as f:
            for in_task_id, eval_task in enumerate(train_task):
                for line in eval_task:
                    f.write(str(line) + '\n')
                accuracies = [r['accuracy'] for r in eval_task]
                score = np.mean(accuracies[sequence_length + 1:])
                iid_score = np.mean(accuracies[1:sequence_length+1])
                f_sum.write(f'  task: {eval_task[0]["task"]:<30}| Network score: {score:0.5f} IID score: {iid_score:0.5f}\n')
                print(f'  task: {eval_task[0]["task"]:<30}| Network score: {score:0.5f} IID score: {iid_score:0.5f}')
                exel_data[f'ood_{task_id}_{in_task_id}'] = score
                exel_data[f'iid_{task_id}_{in_task_id}'] = iid_score

    f_sum.close()


#@ save data to pkl & excel file
    import pandas as pd
    exel_data['tasks'] = str(exel_data['tasks'])
    exel_data['hidden_size'] = exel_data['architecture_params']['hidden_size']
    exel_data['memory_cell_size'] = exel_data['architecture_params']['memory_cell_size'] if 'memory_cell_size' in exel_data['architecture_params'].keys() else 0
    exel_data['memory_size'] = exel_data['architecture_params']['memory_size'] if 'memory_size' in exel_data['architecture_params'].keys() else 0
    exel_data.pop('architecture_params', None)
    exel_data['pretune_steps'] = exel_data['pretune']['pretune_steps'] if 'pretune_steps' in exel_data['pretune'].keys() else 0
    exel_data['train_layers'] = str(exel_data['pretune']['train_layers']) if 'train_layers' in exel_data['pretune'].keys() else 0
    exel_data['pretune'] = exel_data['pretune']['tune']
    exel_data['cl_setting'] = str(exel_data['cl_setting'])

    '''
    if os.path.exists(pkl_log):
        with open(pkl_log, 'rb') as f:
            prev_df = pkl.load(f)
            prev_df = pd.concat([prev_df,pd.DataFrame(exel_data, index=[0])], ignore_index = True)
    else:
        prev_df = pd.DataFrame(exel_data, index=[0])

    prev_df.to_excel(exel_log, index=False)
    with open(pkl_log, 'wb') as f:
        pkl.dump(prev_df, f)
    '''

#@ save data to google drive
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive',
    ]
    json_file_name = '../training/nurips-faaa3b0d0ab4.json'
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
    gc = gspread.authorize(credentials)
    spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1ptbw6ZQmu23TyRIB-hEz4k9J_Is4ifSWjgqmnlWWW-I/edit#gid=0'
    # ?????? ?? ????
    doc = gc.open_by_url(spreadsheet_url)
    # ?? ????
    worksheet_objs = doc.worksheets()
    worksheets_list = []
    for worksheet in worksheet_objs:
        worksheets_list.append(worksheet.title)  # ['decompensation', '??2'] get list of sheet name

    sheet_name = check_dict('sheet_name', 'repeat')
    save_data_col = ['seed', 'tasks', 'architecture', 'joint_training', 'fa_dim', 'training_steps', 'cl_setting', 'share_input', 'etc', 'value']
    if not sheet_name in worksheets_list:
        worksheet = doc.add_worksheet(title=sheet_name, rows='1', cols='200')
        worksheet.insert_row([1]+save_data_col, 1)
    else:
        worksheet = doc.worksheet(sheet_name)

    sheet_data = [check_dict('exp_idx', -1)]
    print(exel_data)
    num_task = len(ast.literal_eval(exel_data['tasks']))
    for key_ in save_data_col:
        if not key_ in ['etc', 'value']:
            sheet_data.append(str(exel_data.pop(key_)))
    ood_score = []
    iid_score = []

    for task_id in range(num_task):
        for in_task_id in range(num_task):
            ood_score.append(exel_data.pop(f'ood_{task_id}_{in_task_id}'))
            iid_score.append(exel_data.pop(f'iid_{task_id}_{in_task_id}'))


    sheet_data = sheet_data+[str(exel_data)]+ood_score+['' for i in range(max(10, num_task)**2 -num_task**2)] + iid_score
    row_current = int(worksheet.acell('A1').value)
    worksheet.insert_row(sheet_data, row_current + 1)
    worksheet.update_acell('A1', row_current + 1)


if __name__ == '__main__':
    '''
    dict_ = {}
    dict_['architecture'] = 'tape_rnn'
    main(dict_)
    '''
    arg_dict = {'developing': True, 'share_input': True, 'joint_training': False, 'architecture': 'lstm',
                   'architecture_params': {'hidden_size': 256, },
                   'training_steps': 10_000, 'tasks': ['dual_ep_pc', 'dual_ep_pc'], 'par2':3, 'debug': False}
    main(arg_dict)
    #!app.run(main)