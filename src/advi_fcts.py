import scipy as sc
import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class Model:
    """
        z ~ Normal(0, I)
        w ~ Normal(0, I)
        sigma ~ LogNormal(1, 1)
        alpha ~ InvGamma(1, 1)
    """
    def __init__(self,data_dim, latent_dim, num_datapoints, dataset):
        self.dim = data_dim*latent_dim + latent_dim*num_datapoints + 1 + latent_dim
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.num_datapoints = num_datapoints
        self.x = dataset
        
        # Priors 
        self.alpha = tfd.InverseGamma(concentration=1.0, scale=tf.ones(latent_dim), name="alpha_prior")
        self.sigma = tfd.LogNormal(loc=0.0,scale=1.0, name="sigma_prior")
        self.z =  tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]), scale=tf.ones([latent_dim, num_datapoints]), name="z_prior")

    def params(self, theta):
        assert theta.shape[0] == self.dim
        theta = tf.reshape(theta, [-1])  
        # Extract parameters
        z_size = self.latent_dim* self.num_datapoints
        w_size = self.data_dim*self.latent_dim

        z_flat = theta[:z_size]
        w_flat = theta[z_size:z_size + w_size]
        sigma = theta[z_size + w_size]
        alpha = theta[z_size + w_size + 1:]

        # Reshape z and w
        z = tf.reshape(z_flat, [self.latent_dim,self.num_datapoints])
        w = tf.reshape(w_flat, [self.data_dim, self.latent_dim])
        return sigma, alpha, z, w
    
    def log_joint(self, theta):
        sigma, alpha, z, w = self.params(theta)
        self.w = tfd.Normal(loc=tf.zeros([self.data_dim, self.latent_dim]), scale=sigma * alpha *tf.ones([self.data_dim, self.latent_dim]), name="w_prior")
        self.log_lik = tfd.Normal(loc=tf.matmul(w,z), scale=sigma*tf.ones([self.data_dim, self.num_datapoints]))
        w_log_prior = self.w.log_prob(w)
        z_log_prior = self.z.log_prob(z)
        sigma_log_prior = self.w.log_prob(sigma) 
        log_lik = self.log_lik.log_prob(self.x)
        return  tf.reduce_sum(log_lik) + tf.reduce_sum(w_log_prior) + tf.reduce_sum(z_log_prior) + tf.reduce_sum(sigma_log_prior)
    
class ADVI_algorithm(Model): 
    def __init__(self,data_dim, latent_dim, num_datapoints, dataset,nb_samples, lr): 
        super().__init__(data_dim, latent_dim, num_datapoints, dataset)
        self.nb_samples = nb_samples 
        self.lr = lr 
    

    def gradient_log_joint(self, theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            log_joint_value = self.log_joint(theta)
            grad = tape.gradient(log_joint_value, theta)
        return grad

    def fct_obj(self, nb_samples):
        global mu, omega
        nabla_mu = tf.zeros((self.dim,1))
        nabla_omega = tf.zeros((self.dim,1))
        for m in range(nb_samples):
            eta = tf.random.normal(shape=(self.dim,1))
            theta = tf.linalg.diag(tf.exp(tf.reshape(omega, [-1])))@eta + mu 
            grad_log_joint_eval = self.gradient_log_joint(theta)
            nabla_mu = nabla_mu + grad_log_joint_eval 
            nabla_omega = nabla_omega + grad_log_joint_eval + tf.linalg.diag(tf.exp(tf.reshape(omega, [-1])))@eta + 1
        return nabla_mu/ nb_samples, nabla_omega/nb_samples

    def step_size(i, lr, s, grad, tau=1, alpha=0.1): 
        s = alpha * grad**2 + (1 - alpha) * s
        rho = lr * (i ** (-0.5 + 1e-16)) / (tau + tf.sqrt(s))
        return rho, s

    def elbo_computation(self, nb_samples): 
        global mu, omega
        elbo = 0
        for _ in range(nb_samples):
            eta = tf.random.normal((self.dim,1))
            theta = tf.linalg.diag(tf.exp(tf.reshape(omega, [-1])))@eta + mu
            elbo = elbo + self.log_joint(theta)  
        return elbo/ nb_samples

    def run_ADVI(self): 
        global mu, omega
        i = 1 # Set iteration counter 
        # Parameter initialization 
        mu = tf.Variable(tf.zeros((self.dim,1)))
        omega = tf.Variable(tf.zeros((self.dim, 1))) # Mean-field   

        elbo_evol = []
        thr =  1e-4
        elbo_ = self.elbo_computation(self.nb_samples)
        condition = True

        while condition: 
            nabla_mu, nabla_omega = self.fct_obj(self.nb_samples)        
            # Calculate step-size rho[i]
            if i ==1: 
                s_mu, s_omega = nabla_mu ** 2, nabla_omega ** 2
            rho_mu, s_mu = self.step_size(i, self.lr, s_mu, nabla_mu)
            rho_omega, s_omega = self.step_size(i, self.lr, s_omega, nabla_omega)

            # Update mu and w 
            mu = mu + rho_mu*nabla_mu
            omega = omega + rho_omega* nabla_omega

            elbo_new = self.elbo_computation(self.nb_samples)
            elbo_evol.append(elbo_new)
            change_in_ELBO = tf.abs(elbo_ - elbo_new)
            elbo_ =  elbo_new
            # increment iteration counter
            i +=1
            if abs(change_in_ELBO) < thr:
                condition = False 
        return mu.numpy(), omega.numpy()
