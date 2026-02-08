import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig, complex_standard_normal


def make_reduction_op_configs():
    for complex in [True, False]:
        with OperationTestConfig.module_name(
            "reduction-complex" if complex else "reduction-real"
        ):
            for reduction in [jnp.sum, jnp.mean, jnp.var, jnp.std]:
                yield from [
                    OperationTestConfig(
                        reduction,
                        complex_standard_normal((4, 5), complex),
                    ),
                    # Explicit argument because capture doesn't work.
                    OperationTestConfig(
                        lambda x, reduction=reduction: reduction(x, axis=1),
                        complex_standard_normal((4, 5), complex),
                    ),
                ]

        with OperationTestConfig.module_name("reduction-real"):
            yield from [
                OperationTestConfig(
                    lambda x: jnp.sum(x, axis=(1, 3, 4)),
                    numpy.random.standard_normal((2, 3, 4, 5, 6)).astype(numpy.float32),
                ),
                OperationTestConfig(
                    lambda x: jnp.max(x, axis=0),
                    numpy.random.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.min(x, axis=-1),
                    numpy.random.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        -jnp.inf,
                        lax.max,
                        window_dimensions=(1, 2, 2),
                        window_strides=(1, 3, 2),
                        padding=((0, 0), (0, 0), (0, 0)),
                    ),
                    numpy.random.standard_normal((2, 6, 6)).astype(numpy.float32),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        0.0,
                        lax.add,
                        window_dimensions=(1, 2, 3, 1),
                        window_strides=(1, 2, 2, 1),
                        padding=((0, 0), (1, 0), (0, 1), (0, 0)),
                    ),
                    numpy.random.standard_normal((2, 5, 6, 3)).astype(numpy.float32),
                    differentiable_argnums=(),
                    name="lax.reduce_window-add-pooling-style",
                ),
                OperationTestConfig(
                    lambda x: jnp.sum(
                        lax.reduce_window(
                            x,
                            -jnp.inf,
                            lax.max,
                            window_dimensions=(1, 2, 2, 1),
                            window_strides=(1, 2, 2, 1),
                            padding=((0, 0), (0, 0), (0, 0), (0, 0)),
                        )
                    ),
                    numpy.random.standard_normal((1, 6, 6, 1)).astype(numpy.float32),
                    name="lax.reduce_window-max-grad-path",
                ),
                OperationTestConfig(
                    lambda x: jnp.sum(
                        lax.reduce_window(
                            x,
                            jnp.inf,
                            lax.min,
                            window_dimensions=(1, 2, 2, 1),
                            window_strides=(1, 2, 2, 1),
                            padding=((0, 0), (0, 0), (0, 0), (0, 0)),
                        )
                    ),
                    numpy.random.standard_normal((1, 6, 6, 1)).astype(numpy.float32),
                    name="lax.reduce_window-min-grad-path",
                ),
                OperationTestConfig(
                    lambda x: jnp.sum(
                        lax.reduce_window(
                            x,
                            -jnp.inf,
                            lax.max,
                            window_dimensions=(1, 2, 2, 2, 1),
                            window_strides=(1, 1, 2, 2, 1),
                            padding=((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
                        )
                    ),
                    numpy.random.standard_normal((1, 4, 6, 6, 1)).astype(numpy.float32),
                    name="lax.reduce_window-max-grad-path-3d",
                ),
            ]
