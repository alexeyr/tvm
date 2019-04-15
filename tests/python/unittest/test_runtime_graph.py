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
import tvm
import numpy as np
import json
from tvm import rpc
from tvm.contrib import util, graph_runtime

def test_graph_simple():
    n = 4
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)

    node0 = {"op": "null", "name": "x", "inputs": []}
    node1 = {"op": "tvm_op", "name": "add",
             "inputs": [[0, 0, 0]],
             "attrs": {"func_name": "myadd",
                       "flatten_data": "1",
                       "num_inputs" : "1",
                    "num_outputs" : "1"}}
    nodes = [node0, node1]
    arg_nodes = [0]
    node_row_ptr = [0, 1, 2]
    outputs = [[1, 0, 0]]
    shape = (4,)
    attrs = {
        "shape" : ["list_shape", [shape, shape]],
        "dltype" : ["list_str", ["float32", "float32"]],
        "storage_id" : ["list_int", [0, 1]],
    }
    graph = {"nodes": nodes,
             "arg_nodes": arg_nodes,
             "node_row_ptr": node_row_ptr,
             "heads": outputs,
             "attrs": attrs}
    graph = json.dumps(graph)

    def check_verify():
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        mod = graph_runtime.create(graph, mlib, tvm.cpu(0))
        a = tvm.testing.random_data(shape=n, dtype=A.dtype)
        mod.run(x=a)
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.asnumpy(), a + 1)

    def check_remote():
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        server = rpc.Server("localhost")
        remote = rpc.connect(server.host, server.port)
        temp = util.tempdir()
        ctx = remote.cpu(0)
        path_dso = temp.relpath("dev_lib.so")
        mlib.export_library(path_dso)
        remote.upload(path_dso)
        mlib = remote.load_module("dev_lib.so")
        mod = graph_runtime.create(graph, mlib, remote.cpu(0))
        a = tvm.testing.random_data(shape=n, dtype=A.dtype)
        mod.run(x=tvm.nd.array(a, ctx))
        out = tvm.nd.empty((n,), ctx=ctx)
        out = mod.get_output(0, out)
        np.testing.assert_equal(out.asnumpy(), a + 1)

    check_verify()
    check_remote()

if __name__ == "__main__":
    test_graph_simple()
