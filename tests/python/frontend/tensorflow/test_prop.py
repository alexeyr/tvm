"""
Tensorflow properties
====================
Property tests for the Tensorflow Relay frontend.
"""
from __future__ import print_function

import sys

import hypothesis.strategies as st
import tensorflow as tf
from hypothesis import *

from .test_forward import _test_pooling
from ...proptest import strategies as tvm_st

#######################################################################
# Pooling
# -------
@st.composite
def pooling_args(draw):
    # TODO TF supports N == 3, the frontend currently doesn't
    N = draw(st.integers(min_value=1, max_value=2))
    input_shape = list(draw(tvm_st.shape(dims=N + 2)))
    window_shape = [draw(st.integers(min_value=1, max_value=input_shape[i + 1])) for i in range(N)]
    padding = draw(st.sampled_from(('SAME', 'VALID')))
    pooling_type = draw(st.sampled_from(('AVG', 'MAX')))
    dilation_or_strides = draw(st.integers(0, 2))
    args = {'input_shape': input_shape, 'window_shape': window_shape,
            'padding': padding, 'pooling_type': pooling_type}
    # pooling with SAME padding is not implemented for dilation_rate > 1
    if dilation_or_strides == 1 and padding != 'SAME':
        def max_dilation(i):
            return input_shape[i + 1] // max(window_shape[i], 1)
        args['dilation_rate'] = [draw(st.integers(min_value=1, max_value=max_dilation(i))) for i in range(N)]
    elif dilation_or_strides == 2:
        # strides > window_shape not supported due to inconsistency between CPU and GPU implementations
        args['strides'] = [draw(st.integers(min_value=1, max_value=window_shape[i])) for i in range(N)]
    return args

@given(pooling_args())
@settings(deadline=500)
def test_forward_pooling(kwargs):
    _test_pooling(**kwargs)

#######################################################################
# Main
# ----
if __name__ == '__main__':
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    glob = globals()
    if test_name:
        tests = {name: glob[name] for name in (test_name, "test_" + test_name, "test_forward_" + test_name) if name in glob and callable(glob[name])}
    else:
        tests = {name: fun for name, fun in glob.items() if name.startswith("test_") and callable(fun)}

    for name, fun in tests.items():
        if name == "test_forward_concat_v2":
            if tf.__version__ == '1.4.1':
                fun()
        else:
            fun()