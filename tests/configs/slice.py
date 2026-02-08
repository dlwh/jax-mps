import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig


def make_slice_op_configs():
    with OperationTestConfig.module_name("slice"):
        return [
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.random.normal(size=(4, 5)),
                (numpy.random.randint(4), numpy.random.randint(5)),
            ),
            OperationTestConfig(
                lambda x, idx, y: x[idx],
                numpy.random.normal(size=(4, 5)),
                (numpy.random.randint(4), numpy.random.randint(5)),
                numpy.asarray(7.0),
            ),
            OperationTestConfig(
                lambda x: lax.dynamic_slice(x, (2,), (4,)),
                numpy.random.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x, idx: jnp.take(x, idx, axis=0),
                numpy.random.normal(size=(5, 3)),
                numpy.array([0, 2, 4]),
            ),
            OperationTestConfig(
                lambda a, ix, iy: a[jnp.arange(a.shape[0]), ix, :, iy],
                numpy.random.normal(size=(2, 4, 6, 5)).astype(numpy.float32),
                numpy.array([0, 2], dtype=numpy.int32),
                numpy.array([1, 3], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda logits, idx: logits[
                    jnp.arange(logits.shape[0])[:, None, None],
                    jnp.arange(logits.shape[1])[None, None, :],
                    idx[:, :, None],
                ],
                numpy.random.normal(size=(2, 3, 5)).astype(numpy.float32),
                numpy.array([[0, 1, 2, 3], [4, 0, 1, 2]], dtype=numpy.int32),
                differentiable_argnums=(),
                name="advanced_gather-new-axis",
            ),
            OperationTestConfig(
                lambda x, idx: jnp.take_along_axis(x, idx, axis=2),
                numpy.random.normal(size=(2, 3, 4, 5)).astype(numpy.float32),
                numpy.random.randint(0, 4, size=(2, 3, 6, 5), dtype=numpy.int32),
                differentiable_argnums=(),
                name="take_along_axis-high-rank-batched",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.random.normal(size=(5, 3)),
                numpy.array([0, 2]),
                numpy.random.normal(size=(2, 3)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].set(1.0),
                numpy.random.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].add(1.0),
                numpy.random.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].divide(2.0),
                numpy.random.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x, update: lax.dynamic_update_slice(x, update, (1, 0)),
                numpy.random.normal(size=(5, 3)),
                numpy.random.normal(size=(2, 3)),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].add(updates),
                numpy.zeros((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.ones((3, 4), dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].subtract(updates),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 0.1, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].mul(updates, unique_indices=True),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 2.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].divide(updates, unique_indices=True),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 2.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].power(updates, unique_indices=True),
                numpy.full((10, 4), 2.0, dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 3.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].min(updates),
                numpy.random.normal(size=(10, 4)).astype(numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.random.normal(size=(3, 4)).astype(numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].max(updates),
                numpy.random.normal(size=(10, 4)).astype(numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.random.normal(size=(3, 4)).astype(numpy.float32),
            ),
            OperationTestConfig(
                lambda x, i0, i1, i2, updates: x.at[i0, i1, i2].add(updates),
                numpy.zeros((2, 3, 5), dtype=numpy.float32),
                numpy.array([[0, 0, 0], [1, 1, 1]], dtype=numpy.int32),
                numpy.array([[0, 1, 2], [0, 1, 2]], dtype=numpy.int32),
                numpy.array([[0, 1, 2], [3, 4, 0]], dtype=numpy.int32),
                numpy.ones((2, 3), dtype=numpy.float32),
                differentiable_argnums=(),
                name="scatter-add-pointwise",
            ),
            OperationTestConfig(
                lambda x, i0, i1, updates: x.at[i0, i1].set(updates),
                numpy.zeros((2, 6), dtype=numpy.float32),
                numpy.array([0, 1], dtype=numpy.int32),
                numpy.array([1, 4], dtype=numpy.int32),
                numpy.array([9.0, 9.0], dtype=numpy.float32),
                differentiable_argnums=(),
                name="scatter-set-pointwise",
            ),
        ]
