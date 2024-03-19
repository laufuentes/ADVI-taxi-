import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn.mixture import BayesianGaussianMixture
tfd = tfp.distributions

from src.advi_fcts import * 
from src.df_processing import * 

def extract_from_VI(mu, omega, advi_model): 
    """ Extracts model parameters from output of PPCA (variational distribution results)

    Args:
        mu (tf.Tensor): mean of variational distribution q
        omega (tf.Tensor): standard deviation of variational distribution q
        advi_model (class): model used for dimension reduction 

    Returns:
        z, w, sigma (tf.Tensors): model parameters
    """
    # Define variational distribution with mu and sigma^2
    # Remark: sigma^2 = exp(omega)
    zeta = np.random.normal(mu, tf.exp(omega))

    # Extract theta from zeta: theta = T^-1(zeta)
    n =  advi_model.latent_dim* advi_model.num_datapoints+advi_model.data_dim*advi_model.latent_dim #position of elements z and w
    first_part = zeta[:n] # identity 
    last_n_elements = tf.exp(zeta[:-n]) # exponential transformation 
 
    theta = tf.reshape(tf.concat([first_part, last_n_elements], axis=0), [-1]) #theta defintion 
    
    # Parameter's dimension
    z_size = advi_model.latent_dim* advi_model.num_datapoints # size of z
    w_size = advi_model.data_dim*advi_model.latent_dim #size of w

    # Extract and reshape parameters
    z = tf.reshape(theta[:z_size], [advi_model.latent_dim,advi_model.num_datapoints])
    w = tf.reshape(theta[z_size:z_size + w_size], [advi_model.data_dim, advi_model.latent_dim])
    sigma = theta[z_size + w_size]
    return z, w, sigma

def perform_BGMM(n_clusters, trajectories, x): 
    """Function that performs Bayesian GMM over a set of reduced dimension trajectories for a predefined number of clusters 

    Args:
        n_clusters (int): number of clusters
        trajectories (np.array): projected trajectories of shape (num_datapoints, latent_dim)
        x (pd.DataFrame): dataset with trajectories and IDs

    Returns:
        _type_: _description_
    """
    bgmm = BayesianGaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)

    # Fit the model to your data
    bgmm.fit(trajectories)

    # Predict cluster memberships for the trajectories
    cluster_memberships = bgmm.predict(trajectories)

    # You can also obtain the posterior probability of each trajectory belonging to each cluster
    posterior_probs = bgmm.predict_proba(trajectories)

    x[str(n_clusters)+'_clusters'] = cluster_memberships
    return x, cluster_memberships, posterior_probs