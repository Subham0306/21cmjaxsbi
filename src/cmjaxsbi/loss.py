import jax
import jax.numpy as jnp


def gaussian_nll(
    y_true,
    means,
    covs,
    weights
):

    """
    y_true:
        (batch, 5)

    means:
        (batch, 2, 5)

    covs:
        (batch, 2, 5, 5)

    weights:
        (batch, 2)
    """

    n_components = means.shape[1]

    log_probs = []

    for k in range(n_components):

        mu = means[:, k]

        C = covs[:, k]

        # Cholesky decomposition

        L = jnp.linalg.cholesky(C)

        # Difference vector

        diff = y_true - mu

        # Mahalanobis term

        solve = jax.scipy.linalg.solve_triangular(
            L,
            diff[..., None],
            lower=True
        )

        maha = jnp.sum(
            solve**2,
            axis=(1, 2)
        )

        # Log determinant

        logdet = 2.0 * jnp.sum(
            jnp.log(
                jnp.diagonal(
                    L,
                    axis1=-2,
                    axis2=-1
                )
            ),
            axis=1
        )

        # Gaussian log-probability

        d = y_true.shape[-1]

        log_prob = -0.5 * (
            maha
            + logdet
            + d * jnp.log(2.0 * jnp.pi)
        )

        log_probs.append(log_prob)

    # Shape = (batch, n_components)

    log_probs = jnp.stack(
        log_probs,
        axis=1
    )

    # Learned mixture weights

    log_weights = jnp.log(
        weights + 1e-12
    )

    # Mixture log probability

    total_log_prob = (
        jax.scipy.special.logsumexp(
            log_probs + log_weights,
            axis=1
        )
    )

    return -jnp.mean(total_log_prob)