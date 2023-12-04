import copy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#static setting
univ_setting = 0 # This indicates it is basic scenario
cuda = 0
cl_setting = str([False, False, 1]).replace(" ", "") #{'cl':True, 'save pre_model make psudo label using it':True, setting:}
fa_dim = 3
pretune = str({'tune': True,'pretune_steps':1_000,'train_layers':['lin_in_','lin_out_'],})
lr_schedule = 0
share_input = 1
developing = 1
sheet_name = 'no_cl'

#dynamic setting
seed_list = [0]
tasks_list = [
    ['cycle_navigation','modular_arithmetic','parity_check','even_pairs',],
    ['even_pairs', 'parity_check', 'modular_arithmetic', 'cycle_navigation', ],
    ]
joint_list = [0]
arch_list = ['rnn', 'lstm', 'stack_rnn', 'tape_rnn',]
step_list = [50000]



task_idx = 0
for idx, tasks in enumerate(tasks_list):
    tasks = ' '.join(tasks)
    for joint in joint_list:
        for architecture in arch_list:
            for training_steps in step_list:
                for seed in seed_list:
                    task_idx += 1
                    exp_idx = copy.deepcopy(task_idx)


                    print('#### Run')
                    print(f"python 0803_all_argpars.py --univ_setting {univ_setting} --exp_idx {exp_idx} --cuda {cuda} "
                          f"--seed {seed} --tasks {tasks} --joint {joint} --architecture {architecture} "
                          f"--training_steps {training_steps} --fa_dim {fa_dim} "
                          f"--share_input {share_input} --developing {developing} --sheet_name {sheet_name} "
                          f"--cl_setting {cl_setting}")
                    os.system(
                        f"python all_argpars.py --univ_setting {univ_setting} --exp_idx {exp_idx} --cuda {cuda} "
                        f"--seed {seed} --tasks {tasks} --joint {joint} --architecture {architecture} "
                        f"--training_steps {training_steps} --fa_dim {fa_dim} --lr_schedule {lr_schedule} "
                        f"--share_input {share_input} --developing {developing} --sheet_name {sheet_name}  "
                        f"--cl_setting {cl_setting}")

'''
parser.add_argument('--univ_setting', default=0,  type=int)
parser.add_argument('--exp_idx', default=-1,  type=int)
parser.add_argument('--cuda', default=0,  type=int)
parser.add_argument('--seed', default=0,  type=int)
parser.add_argument('--tasks', default=str(['even_pairs', 'parity_check', 'cycle_navigation', 'modular_arithmetic', ]),  type=str)
parser.add_argument('--joint', default=bool,  type=False)
parser.add_argument('--cl_setting', default=str((True, True, 1)),  type=str)
parser.add_argument('--architecture', default='lstm',  type=str)
parser.add_argument('--training_steps', default=50000,  type=int)
parser.add_argument('--fa_dim', default=3,  type=int)
parser.add_argument('--pretune', default=str({'tune': True, 'pretune_steps': 1_000, 'train_layers': ['lin_in_','lin_out_'],}),  type=str)
parser.add_argument('--lr_schedule', default=False,  type=bool)
parser.add_argument('--share_input', default=True,  type=bool)
parser.add_argument('--developing', default=True,  type=bool)
'''