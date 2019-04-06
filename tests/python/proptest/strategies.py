import tvm
from hypothesis.strategies import *
import hypothesis.extra.numpy as np_st
from string import ascii_letters


def dtypes(allow_handle=False):
    # float16 seems to produce errors due to precision, may readd later
    number_types = one_of(np_st.integer_dtypes(endianness="="),
                          np_st.unsigned_integer_dtypes(endianness="="),
                          np_st.floating_dtypes(endianness="=", sizes=(32, 64)))
    return number_types | just("handle") if allow_handle else number_types

def shapes(**kwargs):
    dims = kwargs.pop('dims', None)
    if dims:
        kwargs['min_dims'] = kwargs['max_dims'] = dims
    return np_st.array_shapes(**kwargs)

def np_arrays(dtype=dtypes(), shape=shapes(), elements=None, fill=None, unique=False):
    return np_st.arrays(dtype, shape, elements, fill, unique)

def tvm_arrays(**kwargs):
    return np_arrays(**kwargs).map(tvm.ndarray.array)

def var_names():
    return text(alphabet=ascii_letters, min_size=1, max_size=5)

@composite
def tvm_vars(draw, dtype=dtypes()):
    return tvm.var(draw(var_names()), draw(dtype) if isinstance(dtype, SearchStrategy) else dtype)

def positive_integers(max_value=None):
    return integers(min_value=1, max_value=max_value)
