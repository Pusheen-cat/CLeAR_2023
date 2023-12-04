import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from training.Old_rebutal_main_hc import main
import ast
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
dict_ = {}

parser = argparse.ArgumentParser(description='use argparse to run main')


parser.add_argument('--univ_setting', default=0,  type=int)
parser.add_argument('--exp_idx', default=-1,  type=int)
parser.add_argument('--cuda', default=0,  type=int)
parser.add_argument('--seed', default=0,  type=int)
parser.add_argument('--tasks', default=['even_pairs', 'parity_check', 'cycle_navigation', 'modular_arithmetic', ],  nargs='*',type=str)
parser.add_argument('--joint', default=0,  type=int)
parser.add_argument('--architecture', default='lstm',  type=str)
parser.add_argument('--training_steps', default=50000,  type=int)
parser.add_argument('--fa_dim', default=3,  type=int)
parser.add_argument('--pretune', default=str({'tune': True, 'pretune_steps': 1_000, 'train_layers': ['lin_in_','lin_out_'],}),  type=str)
parser.add_argument('--lr_schedule', default=0,  type=int)
parser.add_argument('--share_input', default=1,  type=int)
parser.add_argument('--developing', default=1,  type=int)

parser.add_argument('--dim_match_method', default='mapping',  type=str)
parser.add_argument('--old_cl_method', default='lwf',  type=str)
parser.add_argument('--old_cl_method_parms', default=0,  type=int)

parser.add_argument('--sheet_name', default='none',  type=str)
parser.add_argument('--complexity_parameter_hc', default=[2,3,4,5,6,7,8], nargs='*',type=int)

args = parser.parse_args()

# here change args
dict_['exp_idx'] = args.exp_idx
dict_['univ_setting'] = args.univ_setting # This indicates it is basic scenario
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda)
dict_['seed'] = args.seed
dict_['tasks'] = args.tasks

if args.joint:
    dict_['joint_training'] = True
    dict_['cl_setting'] = (False, False, 1)
else:
    dict_['joint_training'] = False
    dict_['cl_setting'] = (True, args.old_cl_method ,int(args.old_cl_method_parms)) #old_cl_method = 'ewc', 'lwf', 'er'
    '''
    old_cl_method_parms
    lwf: cl_setting[2] 0: softmax (default) / 1: argmax
    ewc: cl_setting[2]: lambda of ewc regularization term
    er: cl_setting[2]: num of memory
    '''
if len(dict_['tasks']) > 4:
    mul_factor = 2
else:
    mul_factor = 1

arch = args.architecture #@
if arch == 'tape_rnn':
    dict_['architecture_params'] = {'hidden_size': 256 * mul_factor, 'memory_cell_size': 8 * mul_factor, 'memory_size': 40}
elif arch == 'stack_rnn':
    dict_['architecture_params'] = {'hidden_size': 256 * mul_factor, 'stack_cell_size': 8 * mul_factor, }
elif arch == 'lstm':
    dict_['architecture_params'] = {'hidden_size': 256 * mul_factor, }
elif arch == 'rnn':
    dict_['architecture_params'] = {'hidden_size': 256 * mul_factor, }
dict_['architecture'] = arch
dict_['training_steps'] = args.training_steps #@
dict_['fa_dim']=args.fa_dim
dict_['pretune'] = ast.literal_eval(args.pretune) #@ #{'tune': True, 'pretune_steps': 1_000, 'train_layers': ['lin_in_','lin_out_'],}

dict_['lr_schedule'] = bool(args.lr_schedule) #False
dict_['share_input'] = bool(args.share_input) #True
dict_['developing'] = bool(args.developing) #True

#@ For old
if args.dim_match_method == 'mapping': #ours
    dict_['use_raw'] = False
    dict_['simple_share'] = False
elif args.dim_match_method == 'embedding':
    dict_['use_raw'] = True
    dict_['simple_share'] = False
    dict_['projection_dim'] = 10
elif args.dim_match_method == 'padding':
    dict_['use_raw'] = True
    dict_['simple_share'] = True

dict_['sheet_name'] = args.sheet_name
dict_['complexity_parameter_hc'] = args.complexity_parameter_hc

main(dict_)