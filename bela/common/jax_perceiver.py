from flax import linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """Simple feed-forward network."""
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x


class Attention(nn.Module):
    """Multi-head attention."""
    dim: int
    num_heads: int

    @nn.compact
    def __call__(self, q, k, v):
        dim_head = self.dim // self.num_heads
        q = nn.Dense(self.dim)(q)
        k = nn.Dense(self.dim)(k)
        v = nn.Dense(self.dim)(v)

        def reshape(x):
            b, t, _ = x.shape
            return x.reshape(b, t, self.num_heads, dim_head)

        q, k, v = map(reshape, (q, k, v))

        attn = jnp.einsum("bthd,bshd->bhts", q, k)
        attn = nn.softmax(attn / jnp.sqrt(dim_head), axis=-1)
        out = jnp.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return nn.Dense(self.dim)(out)


class PerceiverLayer(nn.Module):
    """A single Perceiver layer."""
    dim: int
    num_heads: int
    ff_mult: int = 4
    cross: bool = False

    @nn.compact
    def __call__(self, x, inp=None):
        if self.cross:
            assert inp is not None, "cross attention requires input"
            attn = Attention(self.dim, self.num_heads)(x, inp, inp)
        else:
            attn = Attention(self.dim, self.num_heads)(x, x, x)
        x = x + attn
        x = x + MLP(self.dim, self.dim * self.ff_mult)(x)
        return x


class Perceiver(nn.Module):
    """Minimal Perceiver implementation."""
    dim: int = 512
    depth: int = 6
    num_heads: int = 8
    ff_mult: int = 4
    num_latents: int = 64

    @nn.compact
    def __call__(self, x):
        latents = self.param(
            "latents", nn.initializers.normal(stddev=1.0), (self.num_latents, self.dim)
        )
        lat = jnp.broadcast_to(latents[None], (x.shape[0],) + latents.shape)

        for _ in range(self.depth):
            lat = PerceiverLayer(self.dim, self.num_heads, self.ff_mult, cross=True)(lat, inp=x)
            lat = PerceiverLayer(self.dim, self.num_heads, self.ff_mult)(lat)
        return lat.mean(axis=1)
