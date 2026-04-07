# Could not get working

from functools import partial 

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp 
import numpy as np 

def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y
    

@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)


add_vectors(jnp.arange(8), jnp.arange(8))