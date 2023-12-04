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

"""Constants for the generalization project."""

import functools

import haiku as hk

from models import ndstack_rnn
from models import Old_rnn as rnn
from models import stack_rnn
from models import tape_rnn
from models import transformer

from tasks.cs import FA_binary_addition as binary_addition
from tasks.cs import FA_binary_multiplication as binary_multiplication
from tasks.cs import FA_bucket_sort as bucket_sort
from tasks.cs import FA_compute_sqrt as compute_sqrt
from tasks.cs import FA_duplicate_string as duplicate_string
#from tasks.cs import missing_duplicate_string
from tasks.cs import FA_interlocked_pairing as interlocked_pairing
from tasks.cs import FA_odds_first as odds_first

from tasks.ndcf import FA_divide_by_two as divide_by_two
#from tasks.dcf import modular_arithmetic_brackets
from tasks.dcf import FA_reverse_string as reverse_string
#from tasks.dcf import solve_equation
from tasks.dcf import FA_stack_manipulation as stack_manipulation
from tasks.dcf import FA_compare_occurrence as compare_occurrence

from tasks.regular import FA_cycle_navigation as cycle_navigation
from tasks.regular import FA_even_pairs as even_pairs
from tasks.regular import FA_modular_arithmetic as modular_arithmetic
from tasks.regular import FA_parity_check as parity_check
from training import curriculum as curriculum_lib

#
from tasks.regular import FA_modular_arithmetic_cl as modular_arithmetic_cl
from tasks.regular import FA_cycle_navigation_cl as cycle_navigation_cl
from tasks.dcf import FA_reverse_string_cl as reverse_string_cl
from tasks.cs import FA_duplicate_string_cl as duplicate_string_cl
from tasks.cs import FA_bucket_sort_cl as bucket_sort_cl
#

#this is noisy input generator
from tasks._generate_distribution import FA_noise_distribution as noise_distribution

MODEL_BUILDERS = {
    'rnn':
        functools.partial(rnn.make_rnn, rnn_core=hk.VanillaRNN),
    'lstm':
        functools.partial(rnn.make_rnn, rnn_core=hk.LSTM),
    'stack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=stack_rnn.StackRNNCore,
            inner_core=hk.VanillaRNN),
    'ndstack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=ndstack_rnn.NDStackRNNCore,
            inner_core=hk.VanillaRNN),
    'stack_lstm':
        functools.partial(
            rnn.make_rnn, rnn_core=stack_rnn.StackRNNCore, inner_core=hk.LSTM),
    'transformer_encoder':
        transformer.make_transformer_encoder,
    'transformer':
        transformer.make_transformer,
    'tape_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=tape_rnn.TapeInputLengthJumpCore,
            inner_core=hk.VanillaRNN),
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    # Regular
    'modular_arithmetic':
        modular_arithmetic.ModularArithmetic,
    'parity_check':
        parity_check.ParityCheck,
    'even_pairs':
        even_pairs.EvenPairs,
    'cycle_navigation':
        cycle_navigation.CycleNavigation,

    # Context Free
    'divide_by_two':
        divide_by_two.DivideByTwo,
    'reverse_string':
        reverse_string.ReverseString,
    'stack_manipulation':
        stack_manipulation.StackManipulation,
    'compare_occurrence':
        compare_occurrence.CompareOccurrence,

    # Context Sensitive
    'binary_addition':
        binary_addition.BinaryAddition,
    'binary_multiplication':
        binary_multiplication.BinaryMultiplication,
    'bucket_sort':
        bucket_sort.BucketSort,
    'compute_sqrt':
        compute_sqrt.ComputeSqrt,
    'duplicate_string':
        duplicate_string.DuplicateString,
    'interlocked_pairing':
        interlocked_pairing.InterlockedPairing,
    'odds_first':
        odds_first.OddsFirst,

    ##high-correlation
    'modular_arithmetic_cl':
        modular_arithmetic_cl.ModularArithmetic,
    'cycle_navigation_cl':
        cycle_navigation_cl.CycleNavigation,
    'reverse_string_cl':
        reverse_string_cl.ReverseString,
    'duplicate_string_cl':
        duplicate_string_cl.DuplicateString,
    'bucket_sort_cl':
        bucket_sort_cl.BucketSort,

    # Noise Generating
    'noise_distribution': noise_distribution.NoiseDistribution,
}

'''
'modular_arithmetic_brackets':
    functools.partial(
        modular_arithmetic_brackets.ModularArithmeticBrackets, mult=True),
'''

TASK_LEVELS = {
    'modular_arithmetic': 'regular',
    'parity_check': 'regular',
    'even_pairs': 'regular',
    'cycle_navigation': 'regular',

    'divide_by_two': 'ndcf',
    'reverse_string': 'dcf',
    'stack_manipulation': 'dcf',
    'compare_occurrence': 'dcf',

    'binary_addition': 'cs',
    'binary_multiplication': 'cs',
    'bucket_sort': 'cs',
    'compute_sqrt': 'cs',
    'duplicate_string': 'cs',
    'interlocked_pairing': 'cs',
    'odds_first': 'cs',

    'modular_arithmetic_cl':'regular',
    'cycle_navigation_cl':'regular',
    'reverse_string_cl':'dcf',
    'duplicate_string_cl':'cs',
    'bucket_sort_cl':'cs',

    'noise_distribution':'regular'

}
