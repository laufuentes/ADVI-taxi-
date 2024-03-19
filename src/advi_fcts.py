import scipy as sc
import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.parallel_for.gradients import jacobian

tfd = tfp.distributions


class Model:
    """
        z ~ Normal(0, I)
        w ~ Normal(0, I)
        sigma ~ LogNormal(1, 1)
        alpha ~ InvGamma(1, 1)
    """
    def __init__(self,data_dim, latent_dim, num_datapoints, dataset):
        
        """Parameter initialisation"""

        self.dim = data_dim*latent_dim + latent_dim*num_datapoints + 1 #+ latent_dim
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.num_datapoints = num_datapoints
        self.x = dataset
        
        # Priors 
        #self.alpha = tfd.InverseGamma(concentration=1.0, scale=tf.ones(latent_dim), name="alpha_prior")
        self.sigma = tfd.LogNormal(loc=0.0,scale=1.0, name="sigma_prior")
        self.z =  tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]), scale=tf.ones([latent_dim, num_datapoints]), name="z_prior")

    def params(self, theta):
        """Extracts the samples from $\theta = (z, w, sigma, alpha)$

        Args:
            theta (tf.Tensor): tensor of dimension (self.dim) in R^{self.dim} gathering all parameters to extract (z, w, sigma, alpha) 

        Returns:
            sigma, alpha, z, w (tf.Tensors): extracted parameters
        """
        assert theta.shape[0] == self.dim
        theta = tf.reshape(theta, [-1])  
        # Extract parameters
        z_size = self.latent_dim* self.num_datapoints
        w_size = self.data_dim*self.latent_dim

        z_flat = theta[:z_size]
        w_flat = theta[z_size:z_size + w_size]
        sigma = theta[z_size + w_size]
        #alpha = theta[z_size + w_size + 1:]

        # Reshape z and w
        z = tf.reshape(z_flat, [self.latent_dim,self.num_datapoints])
        w = tf.reshape(w_flat, [self.data_dim, self.latent_dim])
        # alpha
        return sigma, z, w
    
    def log_joint(self, theta):
        """Compute log-joint probability of z, w, alpha, sigma, data according to the precised model 

        Args:
            theta (tf.Tensor): Vector gathering the parameters to extract (z, w, sigma, alpha)

        Returns:
            Value (tf.Tensor): tf.reduce_sum(log_lik) + tf.reduce_sum(w_log_prior) + tf.reduce_sum(z_log_prior) + tf.reduce_sum(sigma_log_prior)
        """
        sigma, z, w = self.params(theta) #alpha
        #alpha = self.alpha.sample()
        # alpha
        self.w = tfd.Normal(loc=tf.zeros([self.data_dim, self.latent_dim]), scale=sigma  *tf.ones([self.data_dim, self.latent_dim]), name="w_prior")
        self.log_lik = tfd.Normal(loc=tf.matmul(w,z), scale=sigma*tf.ones([self.data_dim, self.num_datapoints]))
        w_log_prior = self.w.log_prob(w)
        z_log_prior = self.z.log_prob(z)
        sigma_log_prior = self.sigma.log_prob(sigma) 
        log_lik = self.log_lik.log_prob(self.x)
        return  tf.reduce_sum(log_lik) + tf.reduce_sum(w_log_prior) + tf.reduce_sum(z_log_prior) + tf.reduce_sum(sigma_log_prior)




class ADVI_algorithm(Model): 
    def __init__(self,data_dim, latent_dim, num_datapoints, dataset,nb_samples, lr): 
        super().__init__(data_dim, latent_dim, num_datapoints, dataset)

        """ Parameter initialization"""
        self.nb_samples = nb_samples 
        self.lr = lr 

    def T_inv(self, theta):
        """Computes the inverse transform of zeta, allowing to go from R^{dim} into support of $\theta$. In this case, z_prior and w_prior values are in the same space support as theta, hence no transformation is needed. 
        Regarding alpha_prior and sigma_prior, they return only positive values, hence, computing the probability of alpha_sample<0 or sigma_sample<0 would give a nan. To solve such a problem, 
        we consider for those two parameters the exponential transformation. 

        Args:
            theta (tf.Tensor): Vector gathering the parameters to extract (z, w, sigma, alpha)

        Returns:
            T^1(theta) (tf.Tensor): tf.concat([first_part, last_n_elements], axis=0) has same support as all prior variables 
        """
        n =  self.latent_dim* self.num_datapoints+self.data_dim*self.latent_dim
        first_part = theta[:n]
        last_n_elements = theta[:-n]
        last_n_elements = tf.exp(last_n_elements)
        return tf.concat([first_part, last_n_elements], axis=0)
    
    def Tinv_jac(self, theta):
        """Computes the log absolute value of the determinant of the jacobian matrix of T^(-1)

        Args:
            theta (tf.Tensor): Vector gathering the parameters to extract (z, w, sigma, alpha)

        Returns:
            det (tf.Tensor value): log-absolute value of the determinant of the jacobian of T^(-1)
        """
        n =  self.latent_dim* self.num_datapoints+self.data_dim*self.latent_dim #(dimensions of z and w, where the transformation is the identity)

        mat_id = tf.eye(theta.shape[0]) # Initialize the Jacobian matrix as an identity matrix
        d = tf.concat(( tf.ones((n, 1)),tf.exp(theta[n:])), axis=0)
        jac = d*mat_id #define jacobian
        det = tf.math.log(tf.abs(tf.linalg.det(jac))) #log-abs value of determinant 
        return det
    
    def gradient_Tinv(self, zeta):
        """Compute the gradient of Tinv (T^(-1)) and the log-abs value of determinant of jacobian

        Args:
            zeta (tf.Tensor): Vector gathering the parameters to extract (z, w, sigma, alpha) before transformation (T_inv)

        Returns:
            grad_Tinv (tf.Tensor): gradient of T_inv
            grad_log_jac_Tinv (tf.Tensor): gradient of log-abs value of determinant of jacobian
        """
        with tf.GradientTape() as tape:
            tape.watch(zeta)
            Tinv_value = self.T_inv(zeta)
            grad_Tinv = tape.gradient(Tinv_value, zeta)
        with tf.GradientTape() as tape:
            tape.watch(zeta)
            log_jac_Tinv = self.Tinv_jac(zeta)
            grad_log_jac_Tinv = tape.gradient(log_jac_Tinv, zeta)
        return grad_Tinv, grad_log_jac_Tinv

    def gradient_log_joint(self, theta):
        """Compute the gradient of log-joint probability

        Args:
            theta (tf.Tensor): Vector gathering the parameters to extract (z, w, sigma, alpha)

        Returns:
            grad (tf.Tensor): gradient of log-joint value
        """
        with tf.GradientTape() as tape:
            tape.watch(theta)
            log_joint_value = self.log_joint(theta)
            grad = tape.gradient(log_joint_value, theta)
        return grad

    def fct_obj(self, nb_samples):
        """Computes by Monte Carlo (MC) integration: nabla_mu and nabla_omega (for mu and omega update) and elbo. 
        Args:
            nb_samples (int): Number of samples for MC integration

        Returns:
            nabla_mu/ nb_samples: estimation of nabla_mu by MC integration
            nabla_omega/nb_samples: estimation of nabla_omega by MC integration
            elbo/ nb_samples + entropy: estimation of elbo by MC integration 
        """
        nabla_mu = tf.zeros((self.dim,1))
        nabla_omega = tf.zeros((self.dim,1))
        elbo = 0
        for _ in range(nb_samples):
            eta = tf.random.normal(shape=(self.dim,1))
            par = tf.linalg.diag(tf.exp(tf.reshape(self.omega, [-1])))@eta + self.mu 
            theta = self.T_inv(par)

            grad_log_joint = self.gradient_log_joint(theta)
            grad_Tinv, grad_log_jac_Tinv = self.gradient_Tinv(par)
            nabla_mu = nabla_mu + grad_log_joint * grad_Tinv + grad_log_jac_Tinv 
            nabla_omega = nabla_omega + (grad_log_joint * grad_Tinv + grad_log_jac_Tinv)*tf.linalg.diag(tf.exp(tf.reshape(self.omega, [-1])))@eta  + 1
            elbo = elbo + self.log_joint(theta) + self.Tinv_jac(par)
        entropy = tf.reduce_sum(self.omega)
        return nabla_mu/ nb_samples, nabla_omega/nb_samples, elbo/ nb_samples + entropy

    def step_size(self, i_value, lr, s, grad, tau=1, alpha=0.1): 
        """Computes the step-size for updating mu and omega

        Args:
            i_value (int): iteration number
            lr (float): learning rate
            s (tf.tensor): Value
            grad (tf.tensor): _description_
            tau (int, optional): Defaults to 1.
            alpha (int, optional): Defaults to 0.1.

        Returns:
            rho, s: estimation of step-size parameters
        """
        s = alpha * grad**2 + (1 - alpha) * s
        rho = lr * (i_value ** (-0.5 + 1e-16)) / (tau + tf.sqrt(s))
        return rho, s

    def run_ADVI(self): 
        """Run the whole ADVI algorithm

        Returns:
            self.mu.numpy(): estimated parameter of the distributional distribution: mu 
            self.omega.numpy(): estimated parameter of the distributional distribution: omega 
        """
        i = 1 # Set iteration counter 
        # Parameter initialization 
        self.mu = tf.zeros((self.dim,1))
        self.omega = tf.zeros((self.dim, 1)) # Mean-field   

        self.elbo_evol = []
        thr =  1
        elbo_ = tf.zeros(())
        condition = True

        while condition: 
            # Compute the gradients and new_elbo value
            nabla_mu, nabla_omega, elbo_new = self.fct_obj(self.nb_samples)  
                  
            self.elbo_evol.append(elbo_new)
            change_in_ELBO = elbo_ - elbo_new
            elbo_ =  elbo_new
            # Calculate step-size rho[i]
            if i ==1: 
                s_mu, s_omega = nabla_mu ** 2, nabla_omega ** 2
            rho_mu, s_mu = self.step_size(i, self.lr, s_mu, nabla_mu)
            rho_omega, s_omega = self.step_size(i, self.lr, s_omega, nabla_omega)

            # Update mu and w 
            self.mu = self.mu + tf.linalg.diag(tf.reshape(rho_mu, [-1]))@nabla_mu
            self.omega = self.omega + tf.linalg.diag(tf.reshape(rho_omega, [-1]))@nabla_omega

            if i%10==0: 
                print(i, change_in_ELBO.numpy())
            # increment iteration counter
            i +=1
            if ((np.abs(change_in_ELBO.numpy()) < thr) or (i > 9*1e2) or (np.isnan(elbo_.numpy())) ):
                condition = False 
        return self.mu.numpy(), self.omega.numpy()
