import scipy as sc
import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def substitute_random_variable(rv, value_dict, scope=None):
    """Create a new random variable with substituted values and scoped."""
    with tf.name_scope(scope):
        value = value_dict.get(rv)
        return type(rv)(value=value, name=rv.name)

def create_copy(q, dict_swap, scope=None):
    """Create a copy of the distribution q with scoped name."""
    return substitute_random_variable(q, dict_swap, scope=scope)


def probas(N, M, D):
    ###TODO: verifier que 'on veut vraiment des distributions (avec dimension comprenant les n_samples ou uni-dim)
    with tf.compat.v1.variable_scope(None, default_name="posterior"):
        latent_vars = {}
        # Prior distributions
        # z ~ Multivariate Normal distribution
        z_prior = tfd.MultivariateNormalFullCovariance(loc=tf.zeros([N, M]),covariance_matrix=tf.eye(M))
        latent_vars['z_prior'] = z_prior
        z = z_prior.sample()
        proba_z = tf.reduce_prod(z_prior.prob(z)).numpy() # p_z
        exp_log_proba_z = tf.reduce_prod(tf.exp(z_prior.log_prob(z))) # exp(log_p_z)
        print("p_z: ", proba_z, exp_log_proba_z.numpy())

        # alpha ~ Inverse Gamma distribution
        alpha_prior = tfd.InverseGamma(concentration=1.0, scale=tf.ones(M))
        a = alpha_prior.sample()
        latent_vars['alpha_prior'] = alpha_prior
        proba_alpha = tf.reduce_prod(alpha_prior.prob(a)).numpy()
        print("p_alpha: ", proba_alpha)

        # sigma ~ Log-normal distribution
        sigma_prior = tfd.LogNormal(loc=0.0,scale=1.0)
        latent_vars['sigma_prior'] = sigma_prior
        s = sigma_prior.sample()
        proba_sigma = sigma_prior.prob(s).numpy()
        print("p_sigma: ", proba_sigma)

        # w ~ Multivariate Normal distribution
        w_prior = tfd.MultivariateNormalFullCovariance(loc=tf.zeros([D, M]),covariance_matrix=sigma_prior.sample() * tf.linalg.diag(alpha_prior.sample()))
        latent_vars['w_prior'] = w_prior
        w = w_prior.sample()
        proba_w = tf.reduce_prod(w_prior.prob(w)).numpy()
        print("proba_w: ", proba_w)

        lik = []
        proba_lik = 1
        for i in range(N):
            # Define the multivariate normal distribution
            mvn = tfd.MultivariateNormalDiag(loc=tf.tensordot(w, z[i], axes=1), scale_diag=s * tf.ones([D]))
            
            # Sample from the distribution
            obs = mvn.sample()
            
            # Compute the log probability of the observation under the distribution
            log_prob = mvn.log_prob(obs)
            
            # Append the observation and its log probability
            lik.append(obs)
            proba_lik *= tf.exp(log_prob)

        # Convert proba_lik to a scalar value
        # latent_vars['w_prior'] = w_prior
        proba_lik = tf.squeeze(proba_lik).numpy()
        print("proba_lik: ", proba_lik)
    
    '''  
    print("proba_lik: ",proba_lik)
    priors = [z,alpha,sigma,w, lik]
    proba_priors = [proba_z, proba_alpha, proba_sigma, proba_w, proba_lik]

    #joint distribution 
    p_theta = proba_z*proba_alpha*proba_sigma*proba_w
    jd = p_theta*proba_lik
    print("p_jd: ",jd)'''
    return latent_vars 