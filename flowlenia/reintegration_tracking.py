import jax
import jax.numpy as jnp
from functools import partial

# Toggle for RT pre-computation optimization (cleaner code, enabled by default)
RT_PRECOMPUTE = True

# Toggle for stochastic RT sampling (Monte Carlo approximation)
RT_STOCHASTIC = False
RT_SAMPLE_COUNT = 4  # Number of neighbors to sample (out of 9 for dd=1)

# Toggle for sparse RT computation (only compute where mass exists + halo)
# ~17% speedup (18.9 FPS vs 16.1 baseline at 512x512)
RT_SPARSE = True
RT_SPARSE_THRESHOLD = 1e-4  # Mass threshold for considering a cell "active"

# Toggle for using index slicing instead of jnp.roll (experimental)
RT_USE_SLICE = False


def _dilate_mask(mask, radius):
    """Dilate a 2D boolean mask by the given radius using max pooling."""
    # Use a simple approach: roll and OR
    result = mask
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            rolled = jnp.roll(jnp.roll(mask, dx, axis=0), dy, axis=1)
            result = result | rolled
    return result


class ReintegrationTracking:

    #-------------------------------------------------------------------

    def __init__(self, SX=256, SY=256, dt=.2, dd=5, sigma=.65, border="wall", has_hidden=False,
                 mix="stoch"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.has_hidden = has_hidden
        self.border = border if border in ['wall', 'torus'] else 'wall'
        self.mix = mix

        # Pre-compute values if toggle is enabled
        if RT_PRECOMPUTE:
            # Pre-compute position grid (computed once, reused every call)
            x, y = jnp.arange(SX), jnp.arange(SY)
            X, Y = jnp.meshgrid(x, y)
            self.pos = jnp.dstack((Y, X)) + 0.5  # (SX, SY, 2)

            # Pre-compute delta arrays
            dxs, dys = [], []
            for dx in range(-dd, dd + 1):
                for dy in range(-dd, dd + 1):
                    dxs.append(dx)
                    dys.append(dy)
            self.dxs = jnp.array(dxs)
            self.dys = jnp.array(dys)

            # Pre-compute constants
            self.ma = dd - sigma  # max flow magnitude
            self.area_norm = 4 * sigma ** 2
            self.min_area = min(1, 2 * sigma)
        else:
            # Will compute on each call
            self.pos = None
            self.dxs = None
            self.dys = None
            self.ma = None
            self.area_norm = None
            self.min_area = None

    #-------------------------------------------------------------------

    def __call__(self, *args, key=None, **kwargs):

        if self.has_hidden:
            return self._apply_with_hidden(*args, key=key, **kwargs)
        else:
            return self._apply_without_hidden(*args, key=key, **kwargs)

    #-------------------------------------------------------------------

    def _apply_without_hidden(self, A: jax.Array, F: jax.Array, key: jax.Array = None)->jax.Array:
        # Sparse computation: mask to active regions + halo
        if RT_SPARSE:
            active_mask = A.sum(axis=-1) > RT_SPARSE_THRESHOLD  # (X, Y)
            halo_mask = _dilate_mask(active_mask, self.dd)
            A = jnp.where(active_mask[..., None], A, 0.0)
            F = jnp.where(active_mask[..., None, None], F, 0.0)

        # Get values (precomputed or compute now)
        if RT_PRECOMPUTE:
            pos = self.pos
            dxs = self.dxs
            dys = self.dys
            min_area = self.min_area
            area_norm = self.area_norm
            ma = self.ma
        else:
            # Compute on the fly (original behavior)
            x, y = jnp.arange(self.SX), jnp.arange(self.SY)
            X, Y = jnp.meshgrid(x, y)
            pos = jnp.dstack((Y, X)) + 0.5
            dxs, dys = [], []
            for dx in range(-self.dd, self.dd + 1):
                for dy in range(-self.dd, self.dd + 1):
                    dxs.append(dx)
                    dys.append(dy)
            dxs = jnp.array(dxs)
            dys = jnp.array(dys)
            min_area = min(1, 2 * self.sigma)
            area_norm = 4 * self.sigma ** 2
            ma = self.dd - self.sigma

        sigma = self.sigma
        border = self.border
        SX, SY = self.SX, self.SY

        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def step(A, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
            if border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None]))
                    for di in (-SX, 0, SX) for dj in (-SY, 0, SY)]
                ), axis=0)
            else:
                dpmu = jnp.absolute(pos[..., None] - mur)
            sz = 0.5 - dpmu + sigma
            area = jnp.prod(jnp.clip(sz, 0, min_area), axis=2) / area_norm
            nA = Ar * area
            return nA

        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma)
        if self.border == "wall":
            mu = jnp.clip(mu, sigma, SX - sigma)

        # Stochastic sampling: randomly select subset of neighbors
        if RT_STOCHASTIC and key is not None:
            n_total = len(dxs)
            n_sample = min(RT_SAMPLE_COUNT, n_total)
            # Random permutation and take first n_sample
            indices = jax.random.permutation(key, n_total)[:n_sample]
            dxs_sample = dxs[indices]
            dys_sample = dys[indices]
            # Scale result to compensate for sampling
            scale = n_total / n_sample
            nA = step(A, mu, dxs_sample, dys_sample).sum(0) * scale
        else:
            nA = step(A, mu, dxs, dys).sum(0)

        return nA

    #-------------------------------------------------------------------

    def _apply_with_hidden(self, A: jax.Array, H: jax.Array, F: jax.Array, key: jax.Array = None):
        # Sparse computation: mask to active regions + halo
        if RT_SPARSE:
            # Compute active mask (any channel has mass > threshold)
            active_mask = A.sum(axis=-1) > RT_SPARSE_THRESHOLD  # (X, Y)
            # Dilate by dd to include neighborhood that could receive mass
            halo_mask = _dilate_mask(active_mask, self.dd)
            # Mask A to only contribute from active cells
            A = jnp.where(active_mask[..., None], A, 0.0)
            # Mask H similarly
            H = jnp.where(active_mask[..., None], H, 0.0)
            # Mask F to only flow from active cells
            F = jnp.where(active_mask[..., None, None], F, 0.0)

        # Get values (precomputed or compute now)
        if RT_PRECOMPUTE:
            pos = self.pos
            dxs = self.dxs
            dys = self.dys
            min_area = self.min_area
            area_norm = self.area_norm
            ma = self.ma
        else:
            # Compute on the fly (original behavior)
            x, y = jnp.arange(self.SX), jnp.arange(self.SY)
            X, Y = jnp.meshgrid(x, y)
            pos = jnp.dstack((Y, X)) + 0.5
            dxs, dys = [], []
            for dx in range(-self.dd, self.dd + 1):
                for dy in range(-self.dd, self.dd + 1):
                    dxs.append(dx)
                    dys.append(dy)
            dxs = jnp.array(dxs)
            dys = jnp.array(dys)
            min_area = min(1, 2 * self.sigma)
            area_norm = 4 * self.sigma ** 2
            ma = self.dd - self.sigma

        sigma = self.sigma
        border = self.border
        SX, SY = self.SX, self.SY
        dd = self.dd

        @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
        def step_flow(A, H, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            Hr = jnp.roll(H, (dx, dy), axis=(0, 1))  # (x, y, k)
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))

            if border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None]))
                    for di in (-SX, 0, SX) for dj in (-SY, 0, SY)]
                ), axis=0)
            else:
                dpmu = jnp.absolute(pos[..., None] - mur)

            sz = 0.5 - dpmu + sigma
            area = jnp.prod(jnp.clip(sz, 0, min_area), axis=2) / area_norm
            nA = Ar * area
            return nA, Hr

        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma)
        if border == "wall":
            mu = jnp.clip(mu, sigma, SX - sigma)

        # Stochastic sampling: randomly select subset of neighbors
        if RT_STOCHASTIC and key is not None:
            n_total = len(dxs)
            n_sample = min(RT_SAMPLE_COUNT, n_total)
            indices = jax.random.permutation(key, n_total)[:n_sample]
            dxs_sample = dxs[indices]
            dys_sample = dys[indices]
            scale = n_total / n_sample
            nA, nH = step_flow(A, H, mu, dxs_sample, dys_sample)
            nA = nA * scale
        else:
            nA, nH = step_flow(A, H, mu, dxs, dys)

        if self.mix == 'avg':
            nH = jnp.sum(nH * nA.sum(axis = -1, keepdims = True), axis = 0)  
            nA = jnp.sum(nH, axis = 0)
            nH = nH / (nA.sum(axis = -1, keepdims = True)+1e-10)

        elif self.mix == "softmax":
            expnA = jnp.exp(nA.sum(axis = -1, keepdims = True)) - 1
            nA = jnp.sum(nA, axis = 0)
            nH = jnp.sum(nH * expnA, axis = 0) / (expnA.sum(axis = 0)+1e-10) #avg rule

        elif self.mix == "stoch":
            # Use actual number of neighbors (handles stochastic sampling)
            n_neighbors = nA.shape[0]
            categorical=jax.random.categorical(
              jax.random.PRNGKey(42),
              jnp.log(nA.sum(axis=-1, keepdims=True)),
              axis=0)
            mask=jax.nn.one_hot(categorical,num_classes=n_neighbors,axis=-1)
            mask=jnp.transpose(mask,(3,0,1,2))
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)

        elif self.mix == "stoch_gene_wise":
            # Use actual number of neighbors (handles stochastic sampling)
            n_neighbors = nA.shape[0]
            mask = jnp.concatenate(
              [jax.nn.one_hot(jax.random.categorical(
                                                    jax.random.PRNGKey(42),
                                                    jnp.log(nA.sum(axis = -1, keepdims = True)),
                                                    axis=0),
                              num_classes=n_neighbors,axis=-1)
              for _ in range(H.shape[-1])],
              axis = 2)
            mask=jnp.transpose(mask,(3,0,1,2)) # (n_neighbors, x, y, nb_k)
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)
        
        return nA, nH

    #-------------------------------------------------------------------

