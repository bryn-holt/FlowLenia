import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

ker_f = lambda x, a, w, b : (b * jnp.exp( - (x[..., None] - a)**2 / w)).sum(-1)

bell = lambda x, m, s: jnp.exp(-((x-m)/s)**2 / 2)

def growth(U, m, s):
    return bell(U, m, s)*2-1

kx = jnp.array([
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
])
ky = jnp.transpose(kx)

# Vectorized single-channel Sobel
def _sobel_single_channel(a):
    """Compute Sobel gradient for a single 2D channel."""
    gx = jsp.signal.convolve2d(a, kx, mode='same')
    gy = jsp.signal.convolve2d(a, ky, mode='same')
    return jnp.stack([gy, gx], axis=-1)  # (x, y, 2)

@jax.jit
def sobel(A):
    """
    Vectorized Sobel gradient using vmap over channels.
    A : (x, y, c)
    ret : (x, y, 2, c)
    """
    # vmap over last axis (channels), output on last axis
    return jax.vmap(_sobel_single_channel, in_axes=-1, out_axes=-1)(A)

# Keep old functions for backwards compatibility
def sobel_x(A):
    """
    A : (x, y, c)
    ret : (x, y, c)
    """
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], kx, mode = 'same')
                    for c in range(A.shape[-1])])
def sobel_y(A):
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], ky, mode = 'same')
                    for c in range(A.shape[-1])])



def get_kernels_fft(X, Y, k, R, r, a, w, b):

    """Compute kernels and return a dic containing kernels fft
    
    Args:
        params (Params): raw params of the system
    
    Returns:
        CompiledParams: compiled params which can be used as update rule
    """
    mid = X//2
    Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
          ((R+15) * r[k]) for k in range(k) ]  # (x,y,k)
    K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, a[k], w[k], b[k]) 
                    for k, D in zip(range(k), Ds)])
    nK = K / jnp.sum(K, axis=(0,1), keepdims=True)  # Normalize kernels 
    fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))  # Get kernels fft

    return fK



def get_kernels(SX: int, SY: int, nb_k: int, params):
    mid = SX//2
    Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
          ((params['R']+15) * params['r'][k]) for k in range(nb_k) ]  # (x,y,k)
    K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params["a"][k], params["w"][k], params["b"][k]) 
                    for k, D in zip(range(nb_k), Ds)])
    nK = K / jnp.sum(K, axis=(0,1), keepdims=True)
    return nK


def conn_from_matrix(mat):
    C = mat.shape[0]
    c0 = []
    c1 = [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = mat[s, t]
            if n:
                c0 = c0 + [s]*n
                c1[t] = c1[t] + list(range(i, i+n))
            i+=n
    return c0, c1


def conn_from_lists(c0, c1, C):
    return c0, [[i == c1[i] for i in range(len(c0))] for _ in range(C)]
 