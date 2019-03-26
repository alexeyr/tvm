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

from .test_forward import _test_pooling, _test_convolution
from ...proptest import strategies as tvm_st


def max_dilation(i, input_shape, window_shape):
    return input_shape[i + 1] // max(window_shape[i], 1)


def dilations(i, input_shape, window_shape):
    return tvm_st.positive_integers(max_value=max_dilation(i, input_shape, window_shape))


paddings = st.sampled_from(('SAME', 'VALID'))


#######################################################################
# Pooling
# -------
@st.composite
def pooling_args(draw):
    # docs for allowed args: https://www.tensorflow.org/api_docs/python/tf/nn/pool
    # TODO TF supports N == 3, the frontend currently doesn't
    N = draw(st.integers(min_value=1, max_value=2))
    input_shape = list(draw(tvm_st.shape(dims=N + 2)))
    window_shape = [draw(tvm_st.positive_integers(input_shape[i + 1])) for i in range(N)]
    padding = draw(paddings)
    pooling_type = draw(st.sampled_from(('AVG', 'MAX')))
    dilation_or_strides = draw(st.integers(0, 2))
    args = {'input_shape': input_shape, 'window_shape': window_shape,
            'padding': padding, 'pooling_type': pooling_type}
    # pooling with SAME padding is not implemented for dilation_rate > 1
    if dilation_or_strides == 1 and padding != 'SAME':
        args['dilation_rate'] = [draw(dilations(i, input_shape, window_shape)) for i in range(N)]
    elif dilation_or_strides == 2:
        # strides > window_shape not supported due to inconsistency between CPU and GPU implementations
        args['strides'] = [draw(tvm_st.positive_integers(window_shape[i])) for i in range(N)]
    return args


@given(pooling_args())
@settings(deadline=500)
def test_forward_pooling(kwargs):
    _test_pooling(**kwargs)


#######################################################################
# Convolution
# -----------

@st.composite
def convolution_args(draw):
    # docs for allowed args: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    # TODO _test_convolution only tests nn.conv2d, not the more general nn.convolution

    tensor_shape = draw(tvm_st.shape(dims=4))
    filter_shape = draw(tvm_st.shape(dims=4))
    filter_shape[2] = tensor_shape[3]
    strides = [draw(tvm_st.positive_integers(tensor_shape[i])) for i in (1, 2)]
    dilations = [draw(dilations(i, tensor_shape, filter_shape)) for i in (1, 2)]
    padding = draw(paddings)
    args = {'tensor_in_sizes': tensor_shape, 'filter_in_sizes': filter_shape,
            'dilations': dilations, 'strides': strides, 'padding': padding}
    return args


@given(convolution_args())
def test_forward_convolution(kwargs):
    _test_convolution(**kwargs)


#######################################################################
# Main
# ----
if __name__ == '__main__':
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    glob = globals()
    if test_name:
        tests = {name: glob[name] for name in (test_name, "test_" + test_name, "test_forward_" + test_name) if
                 name in glob and callable(glob[name])}
    else:
        tests = {name: fun for name, fun in glob.items() if name.startswith("test_") and callable(fun)}

    for name, fun in tests.items():
        if name == "test_forward_concat_v2":
            if tf.__version__ == '1.4.1':
                fun()
        else:
            fun()
