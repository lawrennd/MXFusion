import mxnet as mx
from .multivariate_normal import MultivariateNormal


class AffineMeanConditionalNormal:
    """
    This class represents a normal distribution where the mean is an affine transformation of another random variable:
    p(y|x) ~ N(y | Ax + b, C)
    """

    def __init__(self, A, b, covariance):
        """

        :param A: Multiplicative mean term
        :type A: mxnet.nd.array
        :param b: Additive mean term
        :type b: mxnet.nd.array
        :param covariance: Covariance matrix
        :type covariance: mxnet.nd.array
        """
        self.A = A
        self.b = b
        self.covariance = covariance


def marginalise_affine_mean_conditional_normal(p_y_x, p_x):
    """
    ..math
        p(y) = \int p(y|x)p(x) dx

    Where p(y|x) is represented by a AffineMeanConditionalNormal class and p(x) is a multivariate normal

    :param p_y_x:
    :type p_y_x: AffineMeanConditionalNormal
    :param p_x:
    :type p_x MultivariateNormal
    """
    F = mx.nd
    tmp = F.linalg.gemm2(p_y_x.A, p_x.covariance)

    covariance = F.linalg.gemm2(tmp, p_y_x.A, transpose_b=True) + p_y_x.covariance
    mean = F.linalg.gemm2(p_y_x.A, F.expand_dims(p_x.mean, -1)) + F.expand_dims(p_y_x.b, -1)
    return MultivariateNormal(mean[:, :, :, 0], covariance)
