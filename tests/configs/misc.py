import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig


def make_misc_op_configs():
    with OperationTestConfig.module_name("misc"):
        return [
            # This tests transfer of data with non-contiguous arrays.
            OperationTestConfig(
                lambda x: x,
                numpy.random.standard_normal((4, 5, 6, 8)).transpose((2, 0, 1, 3)),
            ),
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=-1),
                numpy.random.standard_normal((3, 5, 7)).astype(numpy.float32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                numpy.random.standard_normal((3, 7)).astype(numpy.float32),
                differentiable_argnums=(),
                name="lax.top_k",
            ),
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [lambda y: y + 1, lambda y: y * 2, lambda y: y - 3],
                    x,
                ),
                numpy.int32(1),
                numpy.random.standard_normal((4,)).astype(numpy.float32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 5,
                    lambda state: (state[0] + 1, state[1] + state[0]),
                    (init, init),
                )[1],
                numpy.int32(0),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x: jnp.fft.fft(x),
                (
                    numpy.random.standard_normal((16,))
                    + 1j * numpy.random.standard_normal((16,))
                ).astype(numpy.complex64),
                differentiable_argnums=(),
            ),
        ]
