__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["Manifold", "Euclidean", "Hyperboloid", "Sphere", "PoincareBall", "StereographicallyProjectedSphere"]

import tensorflow as tf
from tensorflow import Tensor
from typing import Any
from modules import euclidean_distance

eps = 1e-8
MAX_NORM = 85
MIN_NORM = 1e-15


def clamp(x: Tensor, min=float("-inf"), max=float("+inf")):
    return tf.clip_by_value(x, min, max)


def sqrt(x: Tensor):
    x = clamp(x, min=1e-9)  # Smaller epsilon due to precision around x=0.
    return tf.sqrt(x)


def cos(x: Tensor) -> Tensor:
    return tf.cos(x)


def cosh(x: Tensor) -> Tensor:
    x = clamp(x, min=-3, max=3)
    return tf.cosh(x)


def acos(x: Tensor):
    x = clamp(x, min=-1+4*eps, max=1-4*eps)
    return tf.acos(x)


def acosh(x: Tensor) -> Tensor:
    x = clamp(x, min=1+eps)
    return tf.acosh(x)


def sin(x: Tensor) -> Tensor:
    return tf.sin(x)


def sinh(x: Tensor) -> Tensor:
    x = clamp(x, min=-3, max=3)
    return tf.sinh(x)


def tan(x: Tensor) -> Tensor:
    x = clamp(x, min=-1.4+4*eps, max=1.4-4*eps)
    return tf.tan(x)


def atan(x):
    return tf.atan(x)


def tanh(x: Tensor) -> Tensor:
    x = clamp(x, -15.0, 15.0)
    return tf.tanh(x)


def atanh(x: Tensor) -> Tensor:
    x = clamp(x, min=-1.+4*eps, max=1.-4*eps)
    # Numerically stable arctanh that never returns NaNs.
    return tf.atanh(x)


def expand_proj_dims(x: Tensor) -> Tensor:
    shape = tf.concat([tf.shape(x)[:-1], [1]], axis=0)
    zeros = tf.zeros(shape, dtype=x.dtype)
    return tf.concat([zeros, x], axis=-1)


def lorentz_product(x: Tensor, y: Tensor, keepdim: bool=False, dim: int=-1) -> Tensor:
    m = tf.multiply(x, y)
    if keepdim:
        ret = tf.reduce_sum(m, axis=dim, keepdims=True) - 2*m[..., 0:1]
    else:
        ret = tf.reduce_sum(m, axis=dim, keepdims=False) - 2 * m[..., 0]
    return ret


def lorentz_norm(x: Tensor, **kwargs: Any) -> Tensor:
    product = lorentz_product(x, x, **kwargs)
    ret = sqrt(product)
    return ret


def e_0(shape) -> Tensor:
    one_shape = tf.concat([shape[:-1], [1]], axis=0)
    zero_shape = tf.concat([shape[:-1], [shape[-1]-1]], axis=0)

    zeros = tf.zeros(one_shape, dtype=tf.float32)
    ones = tf.ones(zero_shape, dtype=tf.float32)
    e = tf.concat([ones, zeros], axis=-1)
    return e


def _lambda_x(x, c, keepdim: bool=False, dim: int=-1):
    deno = tf.reduce_sum(1-c*tf.pow(x, 2), axis=dim, keepdims=keepdim)
    deno = clamp(deno, min=MIN_NORM)
    return tf.div(2.0, deno)


def pm_mobius_add(x, y, c, dim=-1):
    x2 = tf.reduce_sum(tf.pow(x, 2), axis=dim, keepdims=True)
    y2 = tf.reduce_sum(tf.pow(y, 2), axis=dim, keepdims=True)
    xy = tf.reduce_sum(tf.multiply(x, y), axis=dim, keepdims=True)

    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2

    return tf.div(num, clamp(denom, min=MIN_NORM))


def pm_expmap0(u: Tensor, c: Tensor, dim: int=-1):
    sqrt_c = tf.pow(c, 0.5)
    u_norm = tf.norm(u, ord=2, axis=dim, keepdims=True)
    u_norm = clamp(u_norm, min=MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def pm_expmap(x, u, c, dim:int = -1):
    sqrt_c = tf.pow(c, 0.5)
    u_norm = tf.norm(u, ord=2, axis=dim, keepdims=True)
    u_norm = clamp(u_norm, min=MIN_NORM)

    second_term = (
        tanh(sqrt_c / 2 * _lambda_x(x, c, keepdim=True, dim=dim) * u_norm)
        * u
        / (sqrt_c * u_norm)
    )
    gamma_1 = pm_mobius_add(x, second_term, c, dim=dim)
    return gamma_1


def pm_parallel_transport0(y, v, c, dim: int=-1):
    y2 = tf.reduce_sum(tf.pow(y, 2), axis=dim, keepdims=True)
    ret = v * (1 - c * y2)
    ret = clamp(ret, min=MIN_NORM)
    return ret


class Manifold(object):

    @property
    def radius(self) -> Tensor:
        raise NotImplementedError

    def set_radius(self, radius: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def curvature(self) -> Tensor:
        raise NotImplementedError

    def exp_map(self, v: Tensor, x: Tensor=None) -> Tensor:
        # map vector v from the tangent of x space into manifold
        raise NotImplementedError

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        raise NotImplementedError

    def distance(self, x1: Tensor, x2: Tensor):
        raise NotImplementedError

    def mu0(self, shape):
        raise NotImplementedError

    def expand_proj_dims(self, x):
        raise NotImplementedError


class RadiusManifold(Manifold):
    def __init__(self, radius: tf.Variable):
        self._radius = radius

    @property
    def radius(self) -> tf.Variable:
        # torch.clamp(torch.relu(self._radius), min=1e-8, max=1e8)
        return self._radius

    def set_radius(self, radius: Tensor)->Tensor:
        # radius: place_holder_with_default
        return tf.assign(self._radius, radius)

    @property
    def curvature(self) -> Tensor:
        return tf.pow(self.radius, -2)

    def expand_proj_dims(self, x):
        return expand_proj_dims(x)


class Euclidean(Manifold):
    def __init__(self, radius: tf.Variable):
        pass

    @property
    def radius(self) -> Tensor:
        return tf.constant(0.0, dtype=tf.float32)

    def set_radius(self, radius: Tensor):
        return radius

    @property
    def curvature(self) -> Tensor:
        return tf.constant(0.0, dtype=tf.float32)

    def exp_map(self, v: Tensor, x: Tensor=None) -> Tensor:
        # map vector v from the tangent of x space into manifold
        if x is None:
            ret = v
        else:
            ret = x + v
        return ret

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        return v

    def distance(self, x1: Tensor, x2: Tensor):
        return euclidean_distance(x1, x2)

    def mu0(self, shape):
        return tf.zeros(shape, dtype=tf.float32)

    def expand_proj_dims(self, x):
        return x


class Hyperboloid(RadiusManifold):
    def __init__(self, radius: tf.Variable) -> None:
        super().__init__(radius)

    @property
    def curvature(self) -> Tensor:
        return -super().curvature

    def exp_map(self, v: Tensor, x: Tensor=None) -> Tensor:
        # map vector v from the tangent of x space into manifold
        if x is None:
            # x = self.mu0(v.shape, v.device)
            v = v[..., 1:]
            v_norm = tf.norm(v, ord=2, keepdims=True, axis=-1) / self.radius
            v_normed = tf.nn.l2_normalize(v, axis=-1) * self.radius
            ret = tf.concat((cosh(v_norm) * self.radius, sinh(v_norm) * v_normed), axis=-1)
        else:
            v_norm = lorentz_norm(v, keepdim=True) / self.radius
            v_normed = v / v_norm
            ret = cosh(v_norm) * x + sinh(v_norm) * v_normed

        return ret

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        # Reference: P74 in "Mixed-curvature Variational Autoencoders"
        # PT_{mu0 -> dst}(x) = x + <dst, x>_L / (R^2 - <mu0, dst>_L) * (mu0+dst)
        denom = self.radius * (self.radius + dst[..., 0:1])  # lorentz_product(mu0, dst, keepdim=True) which is -dst[0]*radius
        lp = lorentz_product(dst, v, keepdim=True)
        coef = lp / denom
        right = tf.concat((dst[..., 0:1]+self.radius, dst[..., 1:]), axis=-1)  # mu0 + dst

        return v + coef * right

    def distance(self, x1: Tensor, x2: Tensor):
        # Reference: P70 in "Mixed-curvature Variational Autoencoders"
        return self.radius*acosh(self.curvature*lorentz_product(x1, x2))

    def mu0(self, shape):
        return e_0(shape=shape) * self.radius


class Sphere(RadiusManifold):
    def __init__(self, radius: tf.Variable) -> None:
        super().__init__(radius)

    def exp_map(self, v: Tensor, x: Tensor=None) -> Tensor:
        # map vector v from the tangent of x space into manifold
        if x is None:
            # x = self.mu0(v.shape)
            v = v[..., 1:]
            v_norm = tf.norm(v, ord=2, keepdims=True, axis=-1) / self.radius
            v_normed = tf.nn.l2_normalize(v, axis=-1) * self.radius
            ret = tf.concat((cos(v_norm) * self.radius, sin(v_norm) * v_normed), axis=-1)
        else:
            v_norm = tf.norm(v, ord=2, axis=-1, keepdims=True) / self.radius
            v_normed = v / v_norm
            ret = cos(v_norm) * x + sin(v_norm) * v_normed

        return ret

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        # Reference: P87 in "Mixed-curvature Variational Autoencoders"
        tmp = self.radius * (self.radius + dst[..., 0:1])
        coef = tf.div_no_nan(tf.reduce_sum(dst*v, axis=-1, keepdims=True), tmp)
        right = tf.concat((dst[..., 0:1]+self.radius, dst[..., 1:]), axis=-1)
        return v - coef * right

    def distance(self, x1: Tensor, x2: Tensor):
        # Reference: P82 in "Mixed-curvature Variational Autoencoders"
        return self.radius*acos(self.curvature*tf.reduce_sum(x1*x2, axis=-1))

    def mu0(self, shape):
        return e_0(shape=shape) * self.radius


class PoincareBall(RadiusManifold):
    def __init__(self, radius: tf.Variable) -> None:
        super().__init__(radius)

    @property
    def curvature(self) -> Tensor:
        return -super().curvature

    def exp_map(self, v: Tensor, x: Tensor=None) -> Tensor:
        # map vector v from the tangent of x space into manifold
        if x is None:
            ret = pm_expmap0(v, c=-self.curvature)
        else:
            ret = pm_expmap(x, v, c=-self.curvature)

        return ret

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        ret = pm_parallel_transport0(dst, v, c=-self.curvature)
        return ret

    def distance(self, x1: Tensor, x2: Tensor):
        c = -self.curvature
        sqrt_c = sqrt(c)
        mob = tf.norm(pm_mobius_add(-x1, x2, c=c, dim=-1), ord=2, axis=-1)
        arg = sqrt_c * mob
        dist_c = atanh(arg)
        ret = 2*(dist_c/sqrt_c)
        return ret

    def mu0(self, shape):
        return tf.zeros(shape, dtype=tf.float32)


class StereographicallyProjectedSphere(RadiusManifold):
    def __init__(self, radius: tf.Variable) -> None:
        super().__init__(radius)

    @staticmethod
    def _lambda_x_c(x: Tensor, c: Tensor, dim: int=-1, keepdim: bool=True):
        x2 = tf.reduce_sum(tf.pow(x, 2), axis=dim, keepdims=keepdim)
        ret = 2 / (1 + c * x2)
        ret = clamp(ret, min=MIN_NORM)
        return ret

    def exp_map(self, v: Tensor, x: Tensor=None) -> Tensor:
        # map vector v from the tangent of x space into manifold
        if x is None:
            r = clamp(tf.norm(v, ord=2, axis=-1, keepdims=True), min=MIN_NORM) / self.radius
            ret = tan(r) * v / r
        else:
            r = clamp(tf.norm(v, ord=2, axis=-1, keepdims=True), min=MIN_NORM) / self.radius
            c = self.curvature
            arg = r * self._lambda_x_c(x, c) / 2
            rhs = tan(arg) * v / r
            ret = pm_mobius_add(x, rhs, c=-c)
        return ret

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        ret = (2 / self._lambda_x_c(dst, self.curvature, dim=-1)) * v
        return ret

    def distance(self, x1: Tensor, x2: Tensor):
        ret1 = self.projected_distance(x1, x2)
        # ret2 = self.projected_gyro_distance(x1, x2)
        return ret1

    def projected_distance(self, x1: Tensor, x2: Tensor):
        K = self.curvature
        diff = x1 - x2
        normxmy2 = tf.reduce_sum(diff*diff, axis=-1)  # , keepdim=True
        normx2 = tf.reduce_sum(x1 * x1, axis=-1)  # , keepdim=True
        normy2 = tf.reduce_sum(x2 * x2, axis=-1)  # , keepdim=True

        tmp = tf.div_no_nan(normxmy2, (1 + K * normx2) * (1 + K * normy2))
        dist = 1. / sqrt(K) * acos(clamp(1 - 2 * K * tmp, max=1.0))
        return dist

    def projected_gyro_distance(self, x1: Tensor, x2: Tensor):
        K = self.curvature
        sqrt_K = sqrt(K)
        sm = pm_mobius_add(-x1, x2, c=-K)
        normxy = tf.norm(sm, ord=2, axis=-1)  # , keepdim=True
        dist = 2. / sqrt_K * atan(sqrt_K * normxy)
        return dist

    def mu0(self, shape):
        return tf.zeros(shape, dtype=tf.float32)
