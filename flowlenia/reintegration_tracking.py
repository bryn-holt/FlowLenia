import jax
import jax.numpy as jnp
from functools import partial

# Toggle for RT pre-computation optimization
RT_PRECOMPUTE = True

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

    def __call__(self, *args, **kwargs):
        
        if self.has_hidden:
            return self._apply_with_hidden(*args, **kwargs)
        else:
            return self._apply_without_hidden(*args, **kwargs)

    #-------------------------------------------------------------------

    def _apply_without_hidden(self, A: jax.Array, F: jax.Array)->jax.Array:
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

        nA = step(A, mu, dxs, dys).sum(0)

        return nA

    #-------------------------------------------------------------------

    def _apply_with_hidden(self, A: jax.Array, H: jax.Array, F: jax.Array):
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
            categorical=jax.random.categorical(
              jax.random.PRNGKey(42), 
              jnp.log(nA.sum(axis=-1, keepdims=True)), 
              axis=0)
            mask=jax.nn.one_hot(categorical,num_classes=(2*self.dd+1)**2,axis=-1)
            mask=jnp.transpose(mask,(3,0,1,2)) 
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)

        elif self.mix == "stoch_gene_wise":
            mask = jnp.concatenate(
              [jax.nn.one_hot(jax.random.categorical(
                                                    jax.random.PRNGKey(42), 
                                                    jnp.log(nA.sum(axis = -1, keepdims = True)), 
                                                    axis=0),
                              num_classes=(2*dd+1)**2,axis=-1)
              for _ in range(H.shape[-1])], 
              axis = 2)
            mask=jnp.transpose(mask,(3,0,1,2)) # (2dd+1**2, x, y, nb_k)
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)
        
        return nA, nH

    #-------------------------------------------------------------------

