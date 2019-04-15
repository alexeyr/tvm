# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import math
import nnvm
import topi
import topi.testing
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import onnx
from model_zoo import super_resolution, squeezenet1_1, lenet, resnet18_1_0
from onnx import helper, TensorProto

def get_tvm_output(graph_def, input_data, target, ctx, output_shape=None, output_dtype='float32'):
    """ Generic function to execute and get tvm output"""

    sym, params = nnvm.frontend.from_onnx(graph_def)
    target = 'llvm'
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        dtype_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_data[i].shape
            dtype_dict[input_names[i]] = input_data[i].dtype
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}
        dtype_dict = {input_names: input_data.dtype}

    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                             dtype=dtype_dict, params=params)

    ctx = tvm.cpu(0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_names):
            m.set_input(input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.asnumpy()

def get_caffe2_output(model, x, dtype='float32'):
    import caffe2.python.onnx.backend
    prepared_backend = caffe2.python.onnx.backend.prepare(model)
    W = {model.graph.input[0].name: x.astype(dtype)}
    c2_out = prepared_backend.run(W)[0]
    return c2_out


def verify_onnx_forward_impl(graph_file, data_shape, out_shape):
    dtype = 'float32'
    x = tvm.testing.random_data(data_shape, dtype)
    model = onnx.load_model(graph_file)
    c2_out = get_caffe2_output(model, x, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, dtype)
        tvm.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)

def verify_super_resolution_example():
    verify_onnx_forward_impl(super_resolution, (1, 1, 224, 224), (1, 1, 672, 672))

def verify_squeezenet1_1():
    verify_onnx_forward_impl(squeezenet1_1, (1, 3, 224, 224), (1, 1000))

def verify_lenet():
    verify_onnx_forward_impl(lenet, (1, 1, 28, 28), (1, 10))

def verify_resnet18():
    verify_onnx_forward_impl(resnet18_1_0, (1, 3, 224, 224), (1, 1000))


def test_reshape():
    in_shape = (4, 3, 3, 4)
    ref_shape = (3, 4, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['ref_in'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = onnx.TensorProto.INT32,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(int)))
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    graph = helper.make_graph([ref_node, reshape_node],
                              "reshape_test",
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='reshape_test')

    for target, ctx in ctx_list():
        x = tvm.testing.random_data(in_shape, 'int32')
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, 'float32')

    tvm.testing.assert_allclose(ref_shape, tvm_out.shape)

def test_reshape_like():
    in_shape = (4, 3, 3, 4)
    ref_shape = (3, 4, 4, 3)

    ref_array = tvm.testing.random_data(ref_shape, 'float32')
    ref_node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['ref_in'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = onnx.TensorProto.FLOAT,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(float)))
    copy_node = helper.make_node("Identity", ["ref_in"], ["copy_in"])
    reshape_node = helper.make_node("Reshape", ["in", "copy_in"], ["out"])

    graph = helper.make_graph([ref_node, copy_node, reshape_node],
                              "reshape_like_test",
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='reshape_like_test')

    for target, ctx in ctx_list():
        x = tvm.testing.random_data(in_shape, 'float32')
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, 'float32')

    tvm.testing.assert_allclose(ref_shape, tvm_out.shape)

def _test_power_iteration(x_shape, y_shape):
    if isinstance(y_shape, int):
        y_shape = [y_shape]

    x = tvm.testing.random_data(x_shape, 'float32')
    y = tvm.testing.random_data(y_shape, 'float32')

    np_res = np.power(x, y).astype(np.float32)

    res = helper.make_node("Pow", ['x', 'y'], ['out'])

    graph = helper.make_graph([res],
                              'power_test',
                              inputs = [helper.make_tensor_value_info("x",
                                            TensorProto.FLOAT, list(x_shape)),
                                        helper.make_tensor_value_info("y",
                                            TensorProto.FLOAT, list(y_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(np_res.shape))])

    model = helper.make_model(graph, producer_name='power_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, y], target, ctx, np_res.shape)
        tvm.testing.assert_allclose(np_res, tvm_out, rtol=1e-5, atol=1e-5)

def test_power():
    _test_power_iteration((1, 3), (1))
    _test_power_iteration((2, 3), (2, 3))
    _test_power_iteration((2, 3), (1, 3))

def test_squeeze():
    in_shape = (1, 3, 1, 3, 1, 1)
    out_shape = (3, 3)
    y = helper.make_node("Squeeze", ['in'], ['out'], axes=[0, 2, 4, 5])

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = tvm.testing.random_data(in_shape, 'float32')
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    tvm.testing.assert_allclose(out_shape, tvm_out.shape)

def test_unsqueeze():
    in_shape = (3, 3)
    axis = (0, 3, 4)
    out_shape = (1, 3, 3, 1, 1)
    y = helper.make_node("Unsqueeze", ['in'], ['out'], axes=list(axis))

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = tvm.testing.random_data(in_shape, 'float32')
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    tvm.testing.assert_allclose(out_shape, tvm_out.shape)

def verify_gather(in_shape, indices, axis, dtype):
    x = tvm.testing.random_data(in_shape, dtype)
    indices = np.array(indices, dtype="int32")
    out_np = np.take(x, indices, axis=axis)

    y = helper.make_node("Gather", ['in', 'indices'], ['out'], axis=axis)

    graph = helper.make_graph([y],
                              'gather_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape)),
                                        helper.make_tensor_value_info("indices",
                                            TensorProto.INT32, list(indices.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_np.shape))])
    model = helper.make_model(graph, producer_name='gather_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, indices], target, ctx, out_np.shape)
        tvm.testing.assert_allclose(out_np, tvm_out)

def test_gather():
    verify_gather((4,), [1], 0, 'int32')
    verify_gather((1,4), [0], 0, 'int32')
    verify_gather((4,), [[[1,0],[0,1]]], 0, 'float32')
    verify_gather((2,2), [[[1,0],[0,1]]], 1, 'int32')
    verify_gather((3,3,3), [[[1,0]]], -1, 'int32')
    verify_gather((4,3,5,6), [[2,1,0,0]], 0, 'float32')

def _test_slice_iteration(indata, outdata, starts, ends, axes=None):
    if axes:
        y = helper.make_node("Slice", ['in'], ['out'], axes=axes, starts=starts, ends=ends)
    else:
        y = helper.make_node("Slice", ['in'], ['out'], starts=starts, ends=ends)

    graph = helper.make_graph([y],
                              'slice_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name='slice_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, 'float32')

    tvm.testing.assert_allclose(outdata, tvm_out)

def test_slice():
    x = tvm.testing.random_data((20, 10, 5), np.float32)
    _test_slice_iteration(x, x[0:3, 0:10], (0, 0), (3, 10), (0, 1))
    _test_slice_iteration(x, x[:, :, 3:4], (0, 0, 3), (20, 10, 4))
    _test_slice_iteration(x, x[:, 1:1000], (1), (1000), (1))
    _test_slice_iteration(x, x[:, 0:-1], (0), (-1), (1))

def _test_onnx_op_elementwise(inshape, outfunc, npargs, dtype, opname, kwargs, rtol=1e-7, atol=1e-7):
    indata = tvm.testing.random_data(inshape, dtype, -1, 1)
    outdata = outfunc(indata, **npargs)

    y = helper.make_node(opname, ['in'], ['out'], **kwargs)

    graph = helper.make_graph([y],
                              opname+'_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name=opname+'_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, dtype)

    tvm.testing.assert_allclose(outdata, tvm_out, rtol=rtol, atol=atol)

def test_floor():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.floor, {}, 'float32', 'Floor', {})

def test_ceil():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.ceil, {}, 'float32', 'Ceil', {})

def test_clip():
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              np.clip,
                              {'a_min': -1.0, 'a_max': 1.0},
                              'float32',
                              'Clip',
                              {'min': -1.0, 'max': 1.0})

def test_matmul():
    a_shape = (4, 3)
    b_shape = (3, 4)

    a_array = tvm.testing.random_data(a_shape, 'float32')
    b_array = tvm.testing.random_data(b_shape, 'float32')
    out_np = np.matmul(a_array, b_array)

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

    graph = helper.make_graph([mul_node],
                              "matmul_test",
                              inputs = [helper.make_tensor_value_info("a",
                                            TensorProto.FLOAT, list(a_shape)),
                                        helper.make_tensor_value_info("b",
                                            TensorProto.FLOAT, list(b_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_np.shape))])

    model = helper.make_model(graph, producer_name='matmul_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_array, b_array], target, ctx, out_np.shape)
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

def verify_lrn(shape, nsize, dtype, alpha=None, beta=None, bias=None):
    in_array = tvm.testing.random_data(shape, dtype)

    if alpha == None and beta == None and bias==None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        node = onnx.helper.make_node('LRN', inputs=['in'], outputs=['out'], size=nsize)
    else:
        node = onnx.helper.make_node('LRN', inputs=['in'], outputs=['out'], alpha=alpha,
                                     beta=beta, bias=bias, size=nsize)

    graph = helper.make_graph([node],
                              "lrn_test",
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))])
    model = helper.make_model(graph, producer_name='lrn_test')

    def _get_python_lrn():
        square_sum = np.zeros(shape).astype(dtype)
        for n, c, h, w in np.ndindex(in_array.shape):
            square_sum[n, c, h, w] = sum(in_array[n,
                                         max(0, c - int(math.floor((nsize - 1) / 2))): \
                                             min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                         h,
                                         w] ** 2)
        py_out = in_array / ((bias + (alpha / nsize) * square_sum) ** beta)
        return py_out

    for target, ctx in ctx_list():
        new_sym, params = nnvm.frontend.from_onnx(model)

        input_name = model.graph.input[0].name
        shape_dict = {input_name: in_array.shape}
        dtype_dict = {input_name: dtype}
        graph, lib, params = nnvm.compiler.build(new_sym, target,
                                                 shape_dict, dtype_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input(input_name, tvm.nd.array(in_array.astype(dtype)))
        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(shape, dtype))
        py_out = _get_python_lrn()
        tvm.testing.assert_allclose(py_out, tvm_out.asnumpy(), rtol=1e-5, atol=1e-5)

def test_lrn():
    verify_lrn((5, 5, 5, 5), 3, 'float32')
    verify_lrn((5, 5, 5, 5), 3, 'float32', alpha=0.0002, beta=0.5, bias=2.0)

def _test_upsample_nearest():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in'], ['out'], mode='nearest', scales=[1.0, 1.0, 2.0, 2.0])

    in_array = tvm.testing.random_data(in_shape, 'float32')
    out_array = topi.testing.upsampling_python(in_array, scale, "NCHW")

    graph = helper.make_graph([y],
                              'upsample_nearest_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_nearest_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, 'float32')
        tvm.testing.assert_allclose(out_array, tvm_out)

def _test_upsample_bilinear():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in'], ['out'], mode='linear', scales=[1.0, 1.0, 2.0, 2.0])

    in_array = tvm.testing.random_data(in_shape, 'float32')
    out_array = topi.testing.bilinear_resize_python(in_array, (3*scale, 3*scale), "NCHW")

    graph = helper.make_graph([y],
                              'upsample_bilinear_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_bilinear_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, 'float32')
        tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)

def _test_upsample_bilinear_opset9():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in','scales'], ['out'], mode='linear')
    scales=[1.0, 1.0, 2.0, 2.0]
    in_array = tvm.testing.random_data(in_shape, np.float32)
    out_array = topi.testing.bilinear_resize_python(in_array, (3*scale, 3*scale), "NCHW")

    ref_array = np.array(scales)
    ref_node = helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['scales'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = TensorProto.FLOAT,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(float)))

    graph = helper.make_graph([ref_node, y],
                              'upsample_bilinear_opset9_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_bilinear_opset9_test')
    inputs = []
    inputs.append(in_array)

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, inputs, target, ctx, out_shape, 'float32')
        tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)

def test_upsample():
    _test_upsample_nearest()
    _test_upsample_bilinear()
    _test_upsample_bilinear_opset9()

def _test_softmax(inshape, axis):
    opname = 'Softmax'
    indata = tvm.testing.random_data(inshape, 'float32')
    outshape = inshape
    outdata = topi.testing.softmax_python(indata)
    if isinstance(axis, int):
        y = helper.make_node(opname, ['in'], ['out'], axis = axis)
    elif axis is None:
        y = helper.make_node(opname, ['in'], ['out'])

    graph = helper.make_graph([y],
                              opname+'_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name=opname+'_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outshape, 'float32')
        tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)

def test_softmax():
    _test_softmax((1, 10), None)
    _test_softmax((1, 10), 1)

def verify_min(input_dim):
    dtype = 'float32'

    a_np1 = tvm.testing.random_data(input_dim, dtype)
    a_np2 = tvm.testing.random_data(input_dim, dtype)
    a_np3 = tvm.testing.random_data(input_dim, dtype)

    b_np = np.min((a_np1, a_np2, a_np3), axis=0)

    min_node = helper.make_node("Min", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([min_node],
                              "Min_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Min_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_min():
    verify_min((1, 3, 20, 20))
    verify_min((20, 20))

def verify_max(input_dim):
    dtype = 'float32'

    a_np1 = tvm.testing.random_data(input_dim, dtype)
    a_np2 = tvm.testing.random_data(input_dim, dtype)
    a_np3 = tvm.testing.random_data(input_dim, dtype)

    b_np = np.max((a_np1, a_np2, a_np3), axis=0)

    max_node = helper.make_node("Max", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([max_node],
                              "Max_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Max_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_max():
    verify_max((1, 3, 20, 20))
    verify_max((20, 20))

def verify_mean(input_dim):
    dtype = 'float32'

    a_np1 = tvm.testing.random_data(input_dim, dtype)
    a_np2 = tvm.testing.random_data(input_dim, dtype)
    a_np3 = tvm.testing.random_data(input_dim, dtype)

    b_np = np.mean((a_np1, a_np2, a_np3), axis=0)

    mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([mean_node],
                              "Mean_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Mean_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_mean():
    verify_mean((1, 3, 20, 20))
    verify_mean((20, 20))

def verify_hardsigmoid(input_dim, alpha, beta):
    dtype = 'float32'

    a_np1 = tvm.testing.random_data(input_dim, dtype)

    b_np = np.clip(a_np1 * alpha + beta, 0, 1)

    hardsigmoid_node = helper.make_node("HardSigmoid", ["a_np1"], ["out"], alpha=alpha, beta=beta)

    graph = helper.make_graph([hardsigmoid_node],
                              "HardSigmoid_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='HardSigmoid_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_hardsigmoid():
    verify_hardsigmoid((1, 3, 20, 20), 0.5, 0.6)
    verify_hardsigmoid((20, 20), 0.3, 0.4)

def verify_argmin(input_dim, axis=None, keepdims=None):
    def _argmin_numpy(data, axis=0, keepdims=True):
        result = np.argmin(data, axis=axis)
        if (keepdims == 1):
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = tvm.testing.random_data(input_dim, 'int32', -10, 10)
    if keepdims is None and axis is None:
        b_np = _argmin_numpy(a_np1)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'])
    elif axis is None:
        b_np = _argmin_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmin_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis)
    else:
        b_np = _argmin_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis,
                                     keepdims=keepdims)
    graph = helper.make_graph([node],
                              "argmin_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.INT32, list(a_np1.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.INT32, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='argmin_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def verify_argmax(input_dim, axis=None, keepdims=None):
    def _argmax_numpy(data, axis=0, keepdims=True):
        result = np.argmax(data, axis=axis)
        if (keepdims == 1):
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = tvm.testing.random_data(input_dim, 'int32', -10, 10)

    if keepdims is None and axis is None:
        b_np = _argmax_numpy(a_np1)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'])
    elif axis is None:
        b_np = _argmax_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmax_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis)
    else:
        b_np = _argmax_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis,
                                     keepdims=keepdims)

    graph = helper.make_graph([node],
                              "argmax_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.INT32, list(a_np1.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.INT32, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='argmax_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_arg_min_max():
    '''Verify argmin and argmax'''
    verify_argmin([3,4,4])
    verify_argmax([3,4,4])
    verify_argmin([3,4,4], axis=1)
    verify_argmax([3,4,4], axis=0)
    verify_argmin([3,4,4], keepdims=0)
    verify_argmax([3,4,4], keepdims=1)
    for axis in [0,1,2]:
        for keepdims in [True,False]:
            verify_argmin([3,4,4], axis, keepdims)
            verify_argmax([3,4,4], axis, keepdims)

def verify_constantfill(is_shape, input_dim, out_dim, value, dtype, **kwargs):
    input_a = tvm.testing.random_data(input_dim, dtype)
    out = np.empty(shape=out_dim, dtype=dtype)
    out.fill(value)

    if is_shape == True:
        fill_node = helper.make_node("ConstantFill", [], ["out"], shape=input_dim, value=value, **kwargs)
    else:
        fill_node = helper.make_node("ConstantFill", ["input_a"], ["out"], value=value, dtype=dtype, **kwargs)

    graph = helper.make_graph([fill_node],
                              "fill_test",
                              inputs = [helper.make_tensor_value_info("input_a",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out.shape))])

    model = helper.make_model(graph, producer_name='fill_test')

    for target, ctx in ctx_list():
        if is_shape == True:
            tvm_out = get_tvm_output(model, [], target, ctx, out.shape)
        else:
            tvm_out = get_tvm_output(model, [input_a], target, ctx, out.shape)

        tvm.testing.assert_allclose(out, tvm_out, rtol=1e-5, atol=1e-5)

def test_constantfill():
    verify_constantfill(True, (2, 3, 4, 5), (2, 3, 4, 5), 10, 'float32')
    verify_constantfill(False, (2, 3, 4, 5), (2, 3, 4, 5), 10, 'float32')
    verify_constantfill(True, (2, 3, 4, 5), (2, 3, 4, 5, 4, 5, 6), 10, 'float32', extra_shape=(4, 5, 6))


def verify_pad(indata, pads, value=0.0):
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    len_dim = len(pads) // 2
    np_pads = [(pads[i], pads[i+len_dim]) for i in range(len_dim)]
    outdata = np.pad(indata, pad_width=np_pads, mode='constant', constant_values=value)
    #  onnx graph
    node = helper.make_node(
        'Pad',
        inputs=['input'],
        outputs=['output'],
        mode='constant',
        pads=pads,
        value=value
    )
    graph = helper.make_graph([node],
                              'pad_test',
                              inputs = [helper.make_tensor_value_info("input",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("output",
                                            TensorProto.FLOAT, list(outdata.shape))])
    model = helper.make_model(graph, producer_name='pad_test')
    #  tvm result
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, 'float32')
    tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)

def test_pad():
    verify_pad(tvm.testing.random_data((2, 2), np.float32), [0, 1, 0, 0], 0.0)
    verify_pad(tvm.testing.random_data((2, 3), np.float32), [1, 0, 0, 1], 0.0)
    verify_pad(tvm.testing.random_data((3, 2), np.float32), [0, 0, 1, 0], 5.0)

def verify_reduce_x(name, indata, axis, keepdims):
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    if name == 'ReduceMax':
        outdata = np.maximum.reduce(indata, axis=axis, keepdims=keepdims == 1)
    elif name == 'ReduceMin':
        outdata = np.minimum.reduce(indata, axis=axis, keepdims=keepdims == 1)
    elif name == 'ReduceSum':
        outdata = np.sum(indata, axis=axis, keepdims=keepdims == 1)
    elif name == 'ReduceMean':
        outdata = np.mean(indata, axis=axis, keepdims=keepdims == 1)
    else:
        raise Exception('unsupport op: {}'.format(name))
    if len(np.asarray(outdata).shape) == 0:
        outdata = np.asarray([outdata])
    #  onnx graph
    if axis is None:
        node = helper.make_node(name, inputs=['input'], outputs=['output'],
                                keepdims=keepdims)
    else:
        node = helper.make_node(name, inputs=['input'], outputs=['output'],
                                axis=axis, keepdims=keepdims)
    graph = helper.make_graph([node],
                              '{}_test'.format(name),
                              inputs = [helper.make_tensor_value_info("input",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("output",
                                            TensorProto.FLOAT, list(outdata.shape))])
    model = helper.make_model(graph, producer_name='{}_test'.format(name))
    #  tvm result
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, 'float32')
    tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)

def test_reduce_max():
    verify_reduce_x("ReduceMax",
                    tvm.testing.random_data((3, 2, 2), np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceMax",
                    tvm.testing.random_data((3, 2, 3), np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceMax",
                    tvm.testing.random_data((3, 3, 3), np.float32),
                    axis=(1,), keepdims=1)

def test_reduce_min():
    verify_reduce_x("ReduceMin",
                    tvm.testing.random_data((3, 2, 2), np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceMin",
                    tvm.testing.random_data((3, 2, 3), np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceMin",
                    tvm.testing.random_data((3, 3, 3), np.float32),
                    axis=(1,), keepdims=1)

def test_reduce_sum():
    verify_reduce_x("ReduceSum",
                    tvm.testing.random_data((3, 2, 2), np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceSum",
                    tvm.testing.random_data((3, 2, 3), np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceSum",
                    tvm.testing.random_data((3, 3, 3), np.float32),
                    axis=(1,), keepdims=1)

def test_reduce_mean():
    verify_reduce_x("ReduceMean",
                    tvm.testing.random_data((3, 2, 2), np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceMean",
                    tvm.testing.random_data((3, 2, 3), np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceMean",
                    np.random.randn(3, 3, 3).astype(np.float32),
                    axis=(1,), keepdims=1)

def verify_split(indata, outdatas, split, axis=0):
    indata = np.array(indata).astype(np.float32)
    outdatas = [np.array(o).astype(np.float32) for o in outdatas]
    node = helper.make_node(
        'Split',
        inputs=['input'],
        outputs=['output_{}'.format(i) for i in range(len(split))],
        axis=axis,
        split=split
    )
    graph = helper.make_graph([node],
                              'split_test',
                              inputs = [helper.make_tensor_value_info("input",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("output_{}".format(i),
                                            TensorProto.FLOAT, list(outdatas[i].shape))
                                            for i in range(len(split))
                                         ])
    model = helper.make_model(graph, producer_name='split_test')

    for target, ctx in ctx_list():
        output_shape = [o.shape for o in outdatas]
        output_type = ['float32', 'float32', 'float32']
        tvm_out = get_tvm_output(model, indata, target, ctx, output_shape, output_type)
    for o, t in zip(outdatas, tvm_out):
        tvm.testing.assert_allclose(o, t)

def test_split():
    # 1D
    verify_split([1., 2., 3., 4., 5., 6.], [[1., 2.], [3., 4.], [5., 6.]], [2, 2, 2], 0)
    verify_split([1., 2., 3., 4., 5., 6.], [[1., 2.], [3.], [4., 5., 6.]], [2, 1, 3], 0)
    # 2D
    verify_split([[1., 2., 3., 4.], [7., 8., 9., 10.]],
                 [[[1., 2.], [7., 8.]], [[3., 4.], [9., 10.]]], [2, 2], 1)

def test_binary_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_binary_ops(op, x, y, out_np, broadcast=None, rtol=1e-7, atol=1e-7):
        if broadcast is None:
            z = helper.make_node(op, ['in1', 'in2'], ['out'])
        else:
            z = helper.make_node(op, ['in1', 'in2'], ['out'], broadcast=1)
        graph = helper.make_graph([z],
                                   '_test',
                                  inputs = [helper.make_tensor_value_info("in1",
                                                TensorProto.FLOAT, list(in_shape)),
                                            helper.make_tensor_value_info("in2",
                                                TensorProto.FLOAT, list(in_shape))],
                                  outputs = [helper.make_tensor_value_info("out",
                                                TensorProto.FLOAT, list(out_shape))])
        model = helper.make_model(graph, producer_name='_test')
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(model, [x, y], target, ctx)
            tvm.testing.assert_allclose(out_np, tvm_out, rtol=rtol, atol=atol)

    x = tvm.testing.random_data(in_shape, dtype)
    y = tvm.testing.random_data(in_shape, dtype)
    z = tvm.testing.random_data(shape=(3,), dtype=dtype)
    verify_binary_ops("Add",x, y, x + y, broadcast=None)
    verify_binary_ops("Add", x, z,  x + z, broadcast=True)
    verify_binary_ops("Sub", x, y, x - y, broadcast=None)
    verify_binary_ops("Sub", x, z, x - z, broadcast=True)
    verify_binary_ops("Mul",x, y, x * y, broadcast=None)
    verify_binary_ops("Mul", x, z,  x * z, broadcast=True)
    verify_binary_ops("Div", x, y, x / y, broadcast=None, rtol=1e-5, atol=1e-5)
    verify_binary_ops("Div", x, z, x / z, broadcast=True, rtol=1e-5, atol=1e-5)
    verify_binary_ops("Sum", x, y, x + y, broadcast=None)

def test_single_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_single_ops(op, x, out_np, rtol=1e-7, atol=1e-7):
        z = helper.make_node(op, ['in1'], ['out'])
        graph = helper.make_graph([z],
                                   '_test',
                                  inputs = [helper.make_tensor_value_info("in1",
                                                TensorProto.FLOAT, list(in_shape)),],
                                  outputs = [helper.make_tensor_value_info("out",
                                                TensorProto.FLOAT, list(out_shape))])
        model = helper.make_model(graph, producer_name='_test')
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(model, [x], target, ctx)
            tvm.testing.assert_allclose(out_np, tvm_out, rtol=rtol, atol=atol)

    x = tvm.testing.random_data(in_shape, dtype)
    verify_single_ops("Neg",x, -x)
    verify_single_ops("Abs",x, np.abs(x))
    verify_single_ops("Reciprocal",x, 1/x, rtol=1e-5, atol=1e-5)
    verify_single_ops("Sqrt",x, np.sqrt(x), rtol=1e-5, atol=1e-5)
    verify_single_ops("Relu",x, np.maximum(x, 0))
    verify_single_ops("Exp",x, np.exp(x), rtol=1e-5, atol=1e-5)
    verify_single_ops("Log",x, np.log(x), rtol=1e-5, atol=1e-5)
    verify_single_ops("Log",x, np.log(x), rtol=1e-5, atol=1e-5)
    verify_single_ops("Tanh",x, np.tanh(x), rtol=1e-5, atol=1e-5)
    verify_single_ops("Sigmoid",x, 1 / (1 + np.exp(-x)), rtol=1e-5, atol=1e-5)
    verify_single_ops("Softsign",x, x / (1 + np.abs(x)), rtol=1e-5, atol=1e-5)
    verify_single_ops("SoftPlus",x, np.log(1 + np.exp(x)), rtol=1e-5, atol=1e-5)

def test_leaky_relu():
    def leaky_relu_x(x, alpha):
        return np.where(x >= 0, x, x * alpha)
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              leaky_relu_x,
                              {'alpha': 0.25},
                              'float32',
                              'LeakyRelu',
                              {'alpha': 0.25})

def test_elu():
    def elu_x(x, alpha):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              elu_x,
                              {'alpha': 0.25},
                              'float32',
                              'Elu',
                              {'alpha': 0.25})

def test_selu():
    def selu_x(x, alpha, gamma):
        return gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              selu_x,
                              {'alpha': 0.25, 'gamma': 0.3},
                              'float32',
                              'Selu',
                              {'alpha': 0.25, 'gamma': 0.3})

def test_ThresholdedRelu():
    def ThresholdedRelu_x(x, alpha):
        out_np = np.clip(x, alpha, np.inf)
        out_np[out_np == alpha] = 0
        return out_np
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              ThresholdedRelu_x,
                              {'alpha': 0.25},
                              'float32',
                              'ThresholdedRelu',
                              {'alpha': 0.25})

def test_ScaledTanh():
    def ScaledTanh_x(x, alpha, beta):
        return alpha * np.tanh(beta * x)
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              ScaledTanh_x,
                              {'alpha': 0.25, 'beta': 0.3},
                              'float32',
                              'ScaledTanh',
                              {'alpha': 0.25, 'beta': 0.3})

def test_ParametricSoftplus():
    def ParametricSoftplus_x(x, alpha, beta):
        return alpha * np.log(np.exp(beta * x) + 1)
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              ParametricSoftplus_x,
                              {'alpha': 0.25, 'beta': 0.3},
                              'float32',
                              'ParametricSoftplus',
                              {'alpha': 0.25, 'beta': 0.3})

def test_Scale():
    def Scale_x(x, scale):
        return scale * x
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              Scale_x,
                              {'scale': 0.25},
                              'float32',
                              'Scale',
                              {'scale': 0.25})

def test_LogSoftmax():
    _test_onnx_op_elementwise((1, 4),
                              topi.testing.log_softmax_python,
                              {},
                              'float32',
                              'LogSoftmax',
                              {'axis': 1},
                              rtol=1e-5,
                              atol=1e-5)

if __name__ == '__main__':
    # verify_super_resolution_example()
    # verify_squeezenet1_1()
    # verify_lenet()
    verify_resnet18()
    test_reshape()
    test_reshape_like()
    test_power()
    test_squeeze()
    test_unsqueeze()
    test_slice()
    test_floor()
    test_ceil()
    test_clip()
    test_matmul()
    test_gather()
    test_lrn()
    test_upsample()
    test_forward_min()
    test_forward_max()
    test_forward_mean()
    test_forward_hardsigmoid()
    test_forward_arg_min_max()
    test_softmax()
    test_constantfill()
    test_pad()
    test_reduce_max()
    test_reduce_min()
    test_reduce_sum()
    test_reduce_mean()
    test_split()
    test_binary_ops()
    test_single_ops()
    test_leaky_relu()
    test_elu()
    test_selu()
    test_ThresholdedRelu()
    test_ScaledTanh()
    test_ParametricSoftplus()
    test_Scale()
    test_LogSoftmax()
