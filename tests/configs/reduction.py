import numpy
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
                        lambda rng, complex=complex: complex_standard_normal(
                            rng, (4, 5), complex
                        ),
                    ),
                    # Explicit argument because capture doesn't work.
                    OperationTestConfig(
                        lambda x, reduction=reduction: reduction(x, axis=1),
                        lambda rng, complex=complex: complex_standard_normal(
                            rng, (4, 5), complex
                        ),
                    ),
                ]

        with OperationTestConfig.module_name("reduction-real"):
            yield from [
                OperationTestConfig(
                    lambda x: jnp.max(x, axis=0),
                    lambda rng: rng.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.min(x, axis=-1),
                    lambda rng: rng.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.sum(x, axis=(0, 2)),
                    lambda rng: rng.standard_normal((3, 4, 5)),
                ),
                OperationTestConfig(
                    lambda x: jnp.mean(x, axis=(0, 2)),
                    lambda rng: rng.standard_normal((3, 4, 5)),
                ),
                OperationTestConfig(
                    lambda x: jnp.max(x, axis=(0, 2)),
                    lambda rng: rng.standard_normal((3, 4, 5)),
                    differentiable_argnums=(),
                    name="max-axis-0-2",
                ),
                OperationTestConfig(
                    lambda x: jnp.any(x, axis=(0, 2)),
                    numpy.array(
                        [
                            [[True, False], [False, False], [True, True]],
                            [[False, False], [False, True], [False, False]],
                        ],
                        dtype=bool,
                    ),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.all(x, axis=(1, 2)),
                    numpy.array(
                        [
                            [[True, True], [True, True], [True, True]],
                            [[True, False], [True, True], [True, True]],
                        ],
                        dtype=bool,
                    ),
                    differentiable_argnums=(),
                ),
            ]
