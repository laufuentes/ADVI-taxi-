# Variational Bayes: Experiment in Taxi Routes in Porto

## Project: Graphical Models and Discrete Inference Learning, M2 Mathematics and Artificial Intelligence, Paris-Saclay University

### Nafissa Benali, Laura Fuentes and Rita Maatouk

#### 1- Project Overview

Welcome to our project repository for the course "Graphical Models and Discrete Inference Learning"! We are Nafissa Benali, Laura Fuentes, and Rita Maatouk, M2 students in Mathematics and Artificial Intelligence at Paris-Saclay University. For our final project, we've delved into the realm of variational inference, focusing specifically on the paper titled "Automatic Differentiation Variational Inference" [1].

Our primary goal is to understand and implement the Automatic Differentiation Variational Inference (ADVI) algorithm proposed in the paper. This technique automates variational inference for complex probabilistic models, enabling its application to large datasets efficiently.

#### 2- Implementation Details

We have provided a comprehensive report summarizing the key concepts and methodology outlined in the paper. Additionally, we've implemented the ADVI algorithm, allowing us to apply variational inference to a probabilistic version of Principal Component Analysis (PCA) over a real dataset [2].

##### 2.1- Taxi Routes Experiment

Our focus lies on the taxi routes experiment proposed in the paper. By leveraging ADVI, we aim to accelerate variational inference for dimension reduction in this context. This experiment serves as a practical application to showcase the effectiveness and efficiency of the ADVI algorithm. Such experiment over the taxis dataset could be divided into 3 consecutive steps: 

$\underline{\text{STEP 1: }}$ Interpolation

This step involves normalizing taxi trajectories' length into a 50-coordinate space (x, y), corresponding to an average trip length of 13 minutes. This technique mitigates bias introduced by trajectory length during dimension reduction and clustering. The interpolated trajectories are then saved in "df/interpolation", enabling us to proceed to further steps without repeating this process.

To run this step refer to: 
> 1-Interpolation.ipynb


$\underline{\text{STEP 2: }}$ Dimension reduction

In this step, we perform dimension reduction of the interpolated trajectories using Probabilistic PCA (PPCA). PPCA considers the model $x \sim wz + \text{noise}$ and defines prior distributions over $z$, $w$, and $\text{noise}$ ($p(\theta)$), along with the joint distribution $p(x,\theta)$. We've implemented and deployed the ADVI algorithm to find the parameters of the variational distribution $q(\mu, \omega)$ approximating $p(\theta | x)$, enabling the projection of $x$ into a lower dimension $z$.

To run the second step of the algorithm refer to:
> 2-PPCA_ADVI.ipynb

*Remark: The algorithm may take several hours to converge. We've saved our results from this step in "df/results" for continued use in the subsequent steps.*


$\underline{\text{STEP 3: }}$ Cluster trajectories**

The final step involves clustering the trajectories in the lower dimension. To perform this step, we utilized an existing version of Bayesian Gaussian Mixture Model (BGMM) from sklearn. The final images displaying clustering trajectories and ELBO evolution over iterations are saved in "images/". 

To perform this last step refer to:
> 3-Clustering_BGMM.ipynb


##### 3.2- Package requirements

To run this experiment several packages may be required. We propose to create a new conda environment taxi_advi by copy-pasting the following command on the terminal: 

```
# Create a new Conda environment
conda create -n taxi_advi python=3.8

# Activate the Conda environment
conda activate taxi_advi

# Install the required packages
conda install ipykernel=6.29.3 ipython=8.22.1 keras=2.15.0 matplotlib=3.8.3 numpy=1.26.4 pandas=2.2.1 scikit-learn=1.4.1.post1 scipy=1.12.0 seaborn=0.13.2 six=1.16.0 tensorflow=2.15.0 tensorflow-probability=0.23.0
```

#### 4- References

[1] Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M.Blei. 2016.Automatic Differentiation Variational Inference.

[2] Luis Moreira-Matias, Michel Ferreira, Joao Mendes-Moreira, L. L., J. J. . URL [http://CRAN.R-project.org/package=ipred, r package version 0.9-1](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015)
