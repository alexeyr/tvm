from hypothesis import given
import hypothesis.extra.numpy as np_st
import hypothesis.strategies as st
import tvm
from ..proptest import strategies as tvm_st
from collections import namedtuple
import operator as op
import numpy as np

# We test all together: simplify, canonical_simplify, rewrite_simplify

ArithExpr = namedtuple('ArithExpr', 'expr value vars')

def lift(f, *args):
    exprs = []
    values = []
    vars = dict()
    for x in args:
        if isinstance(x, ArithExpr):
            exprs.append(x.expr)
            values.append(x.value)
            vars.update(x.vars)
        else:
            exprs.append(x)
            values.append(x)
    if isinstance(f, tuple):
        f1, f2 = f
    else:
        f1 = f2 = f
    expr = f1(*exprs)
    value = f2(*values)
    return ArithExpr(expr, value, vars)

def arith_expr():
    known_vars = []
    no_vars = dict()

    @st.composite
    def new_var(draw):
        var_name = "x" + str(len(known_vars) + 1)
        dtype = draw(tvm_st.dtypes())
        value = draw(np_st.from_dtype(dtype))
        var = tvm.var(var_name, dtype.name)
        expr = ArithExpr(var, value, {var: value})
        known_vars.append(expr)
        return expr

    @st.composite
    def new_const(draw):
        dtype = draw(tvm_st.dtypes())
        value = draw(np_st.from_dtype(dtype))
        var = tvm.const(value, dtype.name)
        expr = ArithExpr(var, value, no_vars)
        return expr

    def base():
        return st.one_of(new_const(), st.sampled_from(known_vars), new_var())

    @st.composite
    def extend(draw, children):
        # Avoid enumerating all validity problems, just retry
        with np.errstate(all='raise'):
            for _ in range(5):
                try:
                    kind = draw(tvm_st.positive_integers(2))
                    if kind == 1:
                        child = draw(children)
                        # TODO match child.expr.dtype and do sane operations for type
                        un_ops = [op.neg]
                        # op.abs, op.pos are not supported
                        astype_strat = \
                            tvm_st.dtypes().map(lambda dtype: lambda x: x.astype(dtype.name))
                        f = draw(st.sampled_from(un_ops) | astype_strat)
                        return lift(f, child)
                    elif kind == 2:
                        child1 = draw(children)
                        child2 = draw(children)
                        bin_ops = [op.add, op.sub, op.mul, op.floordiv, op.truediv, op.mod,
                                   op.lshift, op.rshift,
                                   (tvm.max, max), (tvm.min, min)]
                        # op.pow is not supported
                        # boolean ops?
                        # comparison ops?
                        f = draw(st.sampled_from(bin_ops))
                        return lift(f, child1, child2)
                except tvm.TVMError:
                    pass
                except TypeError:
                    pass
                except FloatingPointError:
                    pass

        assert False

    return st.recursive(base(), extend)

def check_simplifier(x, simplify, kind):
    expr = x.expr
    vars = x.vars.keys()
    values = x.vars.values()

    simplified = simplify(expr)
    if tvm.ir_pass.Equal(simplified, expr):
        # simplifier had no effect
        return

    try:
        assert simplified.dtype == expr.dtype
        compute_simplified = tvm.compute((1,), lambda _: simplified)
        s = tvm.create_schedule(compute_simplified.op)
        args = [*vars, compute_simplified]
        f = tvm.build(s, args, 'llvm')
        res_placeholder = tvm.nd.array(np.zeros(1, dtype=expr.dtype))
        f(*values, res_placeholder)
        res = res_placeholder.asnumpy()
        tvm.testing.assert_allclose(res, [x.value])
    except Exception as e:
        raise Exception(f"{kind} invalidly simplified {expr} to {simplified}") from e

@given(arith_expr())
def test_canonical_simplify(x):
    check_simplifier(x, tvm.arith.Analyzer().canonical_simplify, "canonical_simplify")

@given(arith_expr())
def test_rewrite_simplify(x):
    check_simplifier(x, tvm.arith.Analyzer().rewrite_simplify, "rewrite_simplify")

@given(arith_expr())
def test_canonical_simplify_pass(x):
    check_simplifier(x, tvm.ir_pass.CanonicalSimplify, "canonical_simplify_pass")
