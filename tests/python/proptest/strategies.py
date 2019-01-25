import tvm
from hypothesis.strategies import *
# from hypothesis._strategies import defines_strategy # TODO useful?
import hypothesis.extra.numpy as hyp_np
from string import ascii_letters

# @defines_strategy
def dtype(allow_handle=False):
    return hyp_np.scalar_dtypes() | just("handle") if allow_handle else hyp_np.scalar_dtypes()

# @defines_strategy
def shape(**kwargs):
    dims = kwargs.pop('dims', None)
    if dims:
        kwargs['min_dims'] = kwargs['max_dims'] = dims
    return hyp_np.array_shapes(**kwargs)

# @defines_strategy
def np_array(dtype=dtype(), shape=shape(), elements=None, fill=None, unique=False):
    return hyp_np.arrays(dtype, shape, elements, fill, unique)

# @defines_strategy
def var_name():
    return text(alphabet=ascii_letters, min_size=1, max_size=5)

# @defines_strategy
def tvm_var(dtype=dtype()):
    return tvm.var(var_name(), dtype)