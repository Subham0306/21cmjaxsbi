from flax import linen as nn
import jax
import jax.numpy as jnp

class NPE(nn.Module):

    hidden_dim: int = 1024
    n_components: int = 2
    n_params: int = 5

    @nn.compact
    def __call__(self, x):

        # =====================================================
        # Shared trunk
        # =====================================================

        x = nn.leaky_relu(
            nn.Dense(self.hidden_dim)(x),
            negative_slope=0.2
        )

        x = nn.leaky_relu(
            nn.Dense(self.hidden_dim)(x),
            negative_slope=0.2
        )

        x = nn.leaky_relu(
            nn.Dense(self.hidden_dim)(x),
            negative_slope=0.2
        )

        # =====================================================
        # Means
        # =====================================================

        means = nn.Dense(
            self.n_components * self.n_params
        )(x)

        means = means.reshape(
            -1,
            self.n_components,
            self.n_params
        )

        # =====================================================
        # Cholesky params
        # =====================================================

        n_tril = (
            self.n_params *
            (self.n_params + 1)
        ) // 2

        chol_params = nn.Dense(
            self.n_components * n_tril
        )(x)

        chol_params = chol_params.reshape(
            -1,
            self.n_components,
            n_tril
        )

        # =====================================================
        # Mixture weights
        # =====================================================

        logits = nn.Dense(
            self.n_components
        )(x)

        weights = jax.nn.softmax(
            logits,
            axis=-1
        )

        # =====================================================
        # Build covariance matrices
        # =====================================================

        batch_size = x.shape[0]

        L = jnp.zeros(
            (
                batch_size,
                self.n_components,
                self.n_params,
                self.n_params
            )
        )

        tril_idx = jnp.tril_indices(
            self.n_params
        )

        L = L.at[
            :,
            :,
            tril_idx[0],
            tril_idx[1]
        ].set(chol_params)

        diag_idx = jnp.arange(
            self.n_params
        )

        diag_vals = (
            jax.nn.softplus(
                L[:, :, diag_idx, diag_idx]
            )
            + 5e-5
        )

        L = L.at[
            :,
            :,
            diag_idx,
            diag_idx
        ].set(diag_vals)

        covs = jnp.matmul(
            L,
            jnp.swapaxes(L, -1, -2)
        )

        covs = (
            covs
            + 5e-5 * jnp.eye(self.n_params)
        )

        return {
            "means": means,
            "covs": covs,
            "weights": weights
        }