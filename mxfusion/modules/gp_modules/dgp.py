# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import arrays_as_samples

from ...common.config import get_default_dtype
from ...components.distributions import GaussianProcess, Normal, ConditionalGaussianProcess
from ...components.distributions.random_gen import MXNetRandomGenerator
from ...components.functions.operators import broadcast_to
from ...components.variables.var_trans import PositiveTransformation
from ...components.variables.variable import Variable
from ...inference.inference_alg import SamplingAlgorithm
from ...inference.variational import VariationalInference
from ...models import Model, Posterior
from ...modules.module import Module
from ...util.customop import make_diagonal
from ...runtime.distributions.multivariate_normal import MultivariateNormalRuntime


def gaussian_variational_expectation(F, y, variance, f_mean, f_var):
    """
    :param F: mx.nd or mx.sym
    :param y: observed y
    :param variance: likelihood variance
    :param f_mean: prediction mean
    :param f_var: prediction variance
    """
    return -0.5 * np.log(2 * np.pi) - 0.5 * F.log(variance) \
           - 0.5 * (F.square(y - f_mean) + f_var) / variance


def dgp_final_layer_samples(F, variables, model, posterior, n_layers, n_samples):
    """
    Draw samples from final layer of dgp

    :param F: mx.nd or mx.sym
    :param variables: variables dictionary
    :param model: model graph
    :param posterior: posterior graph
    :param n_layers: number of layers
    :param n_samples: number of samples to propagate through layers
    :return: (samples, mean, variance)
    """

    from ...runtime.distributions.conditional_normal import marginalise_affine_mean_conditional_normal
    x = variables[model.X]

    samples = x

    for layer in range(n_layers):
        q_u = get_q_u(F, layer, posterior, variables)

        # Get p(f|u) object
        z = variables[getattr(model, 'inducing_inputs_' + str(layer))]
        conditional_gp = getattr(model, 'F_' + str(layer))
        kern = getattr(model, 'kern_' + str(layer))
        kern_params = kern.fetch_parameters(variables)
        mu = variables[getattr(posterior, 'qU_mean_' + str(layer))]
        rv_shape = mu.shape[1:]

        #print('z' + str(layer), z.shape)
        #print('mu' + str(layer), mu.shape)

        samples, z = arrays_as_samples(mx.nd, [samples, z])
        #print('rv_shape' + str(layer),  rv_shape)
        p_f_given_u = conditional_gp.factor.get_conditional_distribution(samples, z, rv_shape[0], 1e-8, **kern_params)

        q_u = q_u.broadcast_to_n_samples(samples.shape[0])

        # Marginalise u
        p_f = marginalise_affine_mean_conditional_normal(p_f_given_u, q_u)

        # Draw samples
        samples = p_f.draw_samples(n_samples, full_cov=False)
        samples = F.transpose(samples, (0, 2, 1))

    return samples, p_f.mean, p_f.covariance


def get_q_u(F, layer, posterior, variables):
    """
    Makes a MultivariateNormal run time distribution for the variational distribution at the inducing points.
    """
    # Make q(u) multivariate normal object
    S_W = variables[getattr(posterior, 'qU_cov_W_' + str(layer))]
    S_diag = variables[getattr(posterior, 'qU_cov_diag_' + str(layer))]
    cov = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)
    mu = variables[getattr(posterior, 'qU_mean_' + str(layer))]
    return MultivariateNormalRuntime(mu, cov)


class DeepGPLogPdf(VariationalInference):
    """
    The inference algorithm for computing the variational lower bound of the stochastic variational Gaussian process
    with Gaussian likelihood.
    """

    def __init__(self, model, posterior, observed, n_layers, layer_input_dims, layer_output_dims, jitter=1e-6,
                 n_samples=10, dtype=None):
        """
        :param model: Model graph
        :param posterior: Posterior graph
        :param observed: List of observed variables
        :param jitter: Jitter for numerical stability, defaults to 1e-6
        """

        super().__init__(model=model, posterior=posterior, observed=observed)

        if dtype is None:
            self.dtype = get_default_dtype()
        else:
            self.dtype = dtype

        self.log_pdf_scaling = 1.
        self.jitter = jitter
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples
        self.layer_input_dims = layer_input_dims
        self.layer_output_dims = layer_output_dims

    def compute(self, F, variables):
        """
        Compute ELBO of model
        :param F:
        :param variables:
        :return:
        """
        y = variables[self.model.Y]

        # Compute model fit term
        _, mu_f, v_f = dgp_final_layer_samples(F, variables, self.model, self.posterior, self.n_layers, self.n_samples)
        noise_var = variables[self.model.noise_var]

        lik = gaussian_variational_expectation(F, y, noise_var, mu_f, v_f)

        # Compute kl term
        kl = self._compute_kl(F, variables)
        print()
        print('lik', lik.mean(axis=0).sum().asnumpy()[0])
        return self.log_pdf_scaling * lik.mean(axis=0).sum() - kl

    def _compute_kl(self, F, variables):
        """
        Compute sum of KL divergences for each layer of DGP
        """
        kl = 0
        for layer in range(self.n_layers):
            # build variational multivariate normal distribution
            q_u = get_q_u(F, layer, self.posterior, variables)

            # build prior multivariate normal distribution
            z = variables[getattr(self.model, 'inducing_inputs_' + str(layer))]
            kern = getattr(self.model, 'kern_' + str(layer))
            kern_params = kern.fetch_parameters(variables)
            gp_dist = getattr(self.model, 'U_' + str(layer))
            rv_shape = (self.layer_output_dims[layer], z.shape[1]) # FIXME
            #print(rv_shape)
            p_u = gp_dist.factor.get_joint_distribution(z, rv_shape, **kern_params)

            # calculate kl for this layer
            kl = kl + q_u.kl_divergence(p_u)
        return kl


class DeepGPMeanVariancePrediction(SamplingAlgorithm):
    """
    Calculates mean and variance of deep GP prediction
    """
    def __init__(self, model, posterior, observed, n_layers, noise_free=True, n_samples=10, jitter=1e-6, dtype=None):

        super().__init__(model=model, observed=observed, extra_graphs=[posterior])
        if dtype is None:
            self.dtype = get_default_dtype()
        else:
            self.dtype = dtype

        self.jitter = jitter
        self.noise_free = noise_free
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples

    def compute(self, F, variables):
        _, mu_f, v_f = dgp_final_layer_samples(F, variables, self.model, self._extra_graphs[0], self.n_layers, self.n_samples)

        # Add likelihood noise if required
        noise_var = variables[self.model.noise_var]
        if not self.noise_free:
            v_f = v_f + noise_var

        mean = mu_f.mean(axis=0)

        variance_of_mean = F.mean(F.square(mu_f - mean), axis=0)
        var = variance_of_mean + v_f.mean(axis=0)

        outcomes = {self.model.Y.uuid: (mean, var)}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class DeepGPForwardSampling(SamplingAlgorithm):
    """
    Calculates mean and variance of deep GP prediction
    """
    def __init__(self, model, posterior, observed, n_layers, noise_free=True, n_samples=10, jitter=1e-6, dtype=None):

        super().__init__(model=model, observed=observed, extra_graphs=[posterior])
        if dtype is None:
            self.dtype = get_default_dtype()
        else:
            self.dtype = dtype
        self.jitter = jitter
        self.noise_free = noise_free
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples

    def compute(self, F, variables):
        samples, _, _ = dgp_final_layer_samples(F, variables, self.model, self.posterior, self.n_layers, self.n_samples)

        # Add likelihood noise if required
        noise_var = variables[self.model.noise_var]
        if not self.noise_free:
            samples = samples + self._rand_gen.sample_normal(0, F.sqrt(noise_var), shape=samples.shape)

        outcomes = {self.model.Y.uuid: samples}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class DeepGPRegression(Module):
    """
    Deep Gaussian process using doubly stochastic variational inference from:
    Doubly Stochastic Variational Inference for Deep Gaussian Processes (Hugh Salimbeni, Marc Deisenroth)
    https://arxiv.org/abs/1705.08933
    """

    def __init__(self, X, kernels, noise_var, inducing_inputs=None,
                 num_inducing=10, mean=None, n_samples=10, dtype=None, ctx=None):
        """
        :param X: Input variable
        :param kernels: List of kernels for each layer
        :param noise_var: Noise variance for likelihood at final layer
        :param inducing_inputs: List of variables that represent the inducing points at each layer or None
        :param num_inducing: Number of inducing points at each layer in inducing_inputs is None
        :param mean: Not used yet
        :param dtype: dtype to use when creating mxnet arrays
        :param ctx: mxnet context
        """

        self.n_layers = len(kernels)

        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)

        self.layer_input_dims = [kern.input_dim for kern in kernels]
        self.layer_output_dims = self.layer_input_dims[1:] + [1]

        if inducing_inputs is None:
            inducing_inputs = [Variable(shape=(num_inducing, self.layer_input_dims[i])) for i in range(self.n_layers)]

        self.inducing_inputs = inducing_inputs
        inducing_inputs_tuples = []

        for i, inducing in enumerate(inducing_inputs):
            inducing_inputs_tuples.append(('inducing_inputs_' + str(i), inducing))

        inputs = [('X', X)] + inducing_inputs_tuples + [('noise_var', noise_var)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super().__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, dtype=dtype, ctx=ctx)
        self.mean_func = mean
        self.kernels = kernels
        self.n_samples = n_samples

    def _generate_outputs(self, output_shapes=None):
        """
        Generate the output of the module with given output_shapes.
        :param output_shapes: the shapes of all the output variables
        :type output_shapes: {str: tuple}
        """
        if output_shapes is None:
            Y_shape = self.X.shape[:-1] + (1,)
        else:
            Y_shape = output_shapes['random_variable']
        self.set_outputs([Variable(shape=Y_shape)])

    def _build_module_graphs(self):
        """
        Generate a model graph for GP regression module.
        """
        Y = self.random_variable

        graph = Model(name='dgp_regression')
        graph.X = self.X.replicate_self()
        graph.noise_var = self.noise_var.replicate_self()

        post = Posterior(graph)
        N = Y.shape[0]

        for i in range(self.n_layers):
            z = self.inducing_inputs[i].replicate_self()
            setattr(graph, 'inducing_inputs_' + str(i), z)

            M = z.shape[0]

            u = GaussianProcess.define_variable(X=z, kernel=self.kernels[i], shape=(M, self.layer_output_dims[i]),
                                                rand_gen=self._rand_gen, dtype=self.dtype,
                                                ctx=self.ctx)
            setattr(graph, 'U_' + str(i), u)
            f = ConditionalGaussianProcess.define_variable(
                X=graph.X, X_cond=z, Y_cond=u,
                kernel=self.kernels[i], shape=(N, self.layer_output_dims[i]), mean=None,
                rand_gen=self._rand_gen, dtype=self.dtype, ctx=self.ctx)
            setattr(graph, 'F_' + str(i), f)

            setattr(graph, 'kern_' + str(i), u.factor.kernel)

            setattr(post, 'qU_cov_diag_' + str(i), Variable(shape=(self.layer_output_dims[i], M,),
                                                            transformation=PositiveTransformation()))
            setattr(post, 'qU_cov_W_' + str(i), Variable(shape=(self.layer_output_dims[i], M, M)))
            setattr(post, 'qU_mean_' + str(i), Variable(shape=(self.layer_output_dims[i], M)))

        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(mean=getattr(graph, 'F_' + str(self.n_layers - 1)),
                                 variance=broadcast_to(graph.noise_var, graph.Y.shape),
                                 rand_gen=self._rand_gen,
                                 dtype=self.dtype, ctx=self.ctx))

        return graph, [post]

    def _attach_default_inference_algorithms(self):
        """
        Attach the default inference algorithms for SVGPRegression Module:
        log_pdf <- DeepGPLogPdf
        prediction <- DeepGPMeanVariancePrediction
        """
        observed = [v for k, v in self.inputs] + \
                   [v for k, v in self.outputs]
        self.attach_log_pdf_algorithms(targets=self.output_names, conditionals=self.input_names,
                                       algorithm=DeepGPLogPdf(self._module_graph, self._extra_graphs[0], observed,
                                                              self.n_layers, self.layer_input_dims,
                                                              self.layer_output_dims, n_samples=self.n_samples,
                                                              dtype=self.dtype), alg_name='dgp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_prediction_algorithms(
           targets=self.output_names, conditionals=self.input_names,
           algorithm=DeepGPMeanVariancePrediction(self._module_graph, self._extra_graphs[0], observed, self.n_layers,
                                                  n_samples=self.n_samples, dtype=self.dtype), alg_name='dgp_predict')

        self.attach_draw_samples_algorithms(
           targets=self.output_names, conditionals=self.input_names,
           algorithm=DeepGPForwardSampling(self._module_graph, self._extra_graphs[0], observed, self.n_layers,
                                           n_samples=self.n_samples, dtype=self.dtype), alg_name='dgp_sample')

    @staticmethod
    def define_variable(X, kernels, noise_var, shape=None, inducing_inputs=None, num_inducing=10, mean_func=None,
                        n_samples=10, dtype=None, ctx=None):
        """
        Creates and returns a variable drawn from a doubly stochastic deep GP
        :param X: Input variable
        :param kernels: List of kernels for each layer
        :param noise_var: Noise variance for likelihood at final layer
        :param shape: Shape of variable
        :param inducing_inputs: List of variables that represent the inducing points at each layer or None
        :param num_inducing: Number of inducing points at each layer in inducing_inputs is None
        :param mean_func: Not used yet
        :param dtype: dtype to use when creating mxnet arrays
        :param ctx: mxnet context
        """
        gp = DeepGPRegression(
            X=X, kernels=kernels, noise_var=noise_var,
            inducing_inputs=inducing_inputs, num_inducing=num_inducing,
            mean=mean_func, n_samples=n_samples, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        rep = super().replicate_self(attribute_map)

        rep.kernels = [k.replicate_self(attribute_map) for k in self.kernels]
        rep.mean_func = None if self.mean_func is None else self.mean_func.replicate_self(attribute_map)
        return rep