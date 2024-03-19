import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from src.clustering import *
from src.advi_fcts import * 
from src.df_processing import * 

def generate_colors(n):
    """Function that generates a palette of n colors for labeling in final figure

    Args:
        n (int): number of clusters for generating a palette

    Returns:
        sns.color_palette("Spectral", n): color palette
    """
    return sns.color_palette("Spectral", n)


def clusters_fig(reshaped, num_datapoints, nb_clusters, colors_nclusters, cm): 
    """Function that produces an image that will be saved in images/

    Args:
        reshaped (np.array): reshaped trajectories (vector format)
        num_datapoints (int): number of datapoints in dataset
        nb_clusters (list of integers): list of 3 elements containing the number of produced clustering
        colors_nclusters (list): list of color palettes for presenting clusters 
        cm (list): list containing cluster memberships for each element in dataset for each predefined number of clusters
    """
    _, ax = plt.subplots(ncols=3, figsize=(20,5))
    for j, i in enumerate(reshaped): 
        r = i.reshape(50,2)

        # 2 clusters
        col2 = colors_nclusters[0][cm[0][j]]
        ax[0].plot(r[:,0], r[:,1], color=col2)
        ax[0].set_title(str(nb_clusters[0])+' Clusters')

        # 15 clusters
        col15 = colors_nclusters[1][cm[1][j]]
        ax[1].plot(r[:,0], r[:,1], color=col15)
        ax[1].set_title(str(nb_clusters[1])+' Clusters')

        # 30 clusters
        col30 = colors_nclusters[2][cm[2][j]]
        ax[2].plot(r[:,0], r[:,1], color=col30)
        ax[2].set_title(str(nb_clusters[2])+' Clusters')

    # Save image
    plt.savefig('images/Clusters_'+str(num_datapoints)+'_trajectories.png')
    plt.show()
    pass

def ELBO_fig(elbo_evol):
    """Function that creates the figure of ELBO evolution

    Args:
        elbo_evol (np.array): array containing ELBO values over iterations
    """
    plt.plot(elbo_evol[:,0], elbo_evol[:,1])
    plt.title('ELBO evolution over iterations')
    plt.xlabel('Number of iterations')
    plt.ylabel('ELBO value')

    plt.savefig('images/ELBO_Evolution.png')
    plt.show()
    pass