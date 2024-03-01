import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc

def extract_traj(df): 
    """
    Extracts the trajectories stored (as strings) in a dataframe. 

    Args:
        df (pd.DataFrame): Dataframe gathering trajectories information

    Returns:
        list_trips (list): list of processed trajectories (as np.array instead of str elements)
    """
    n = np.shape(df)[0]
    list_trips = []
    for i in df['POLYLINE']: 
        list_trips.append(np.array(eval(i)))
    return list_trips

def plot_trajectories_list(ls): 
    """
    Plot all trajectories from a list of coordinates  

    Args:
        ls (list): contains a group of coordinates (x,y)
    """
    for i in range(len(ls)): 
        plt.plot(ls[i][:,0], ls[i][:,1])
    plt.show()
    pass

def interpolation(ls, num_points):
    """
    Linear interpolation of the given data points.
    
    Args:
        x (list): List of x-coordinates.
        y (list): List of y-coordinates.
        num_points (int): Number of points for interpolation.
        
    Returns:
        tuple: Tuple containing the interpolated x and y coordinates.
    """
    newls = []
    for i in range(len(ls)): 
        x = ls[i][:,0]
        y = ls[i][:,1]

        # Create an interpolation function for x and y separately
        f_x = sc.interpolate.interp1d(np.linspace(0, 1, len(x)), x, kind='linear')
        f_y = sc.interpolate.interp1d(np.linspace(0, 1, len(y)), y, kind='linear')
        
        # Interpolate y values
        t = np.linspace(0, 1, 50)
        interpolated_x = f_x(t)
        interpolated_y = f_y(t)

        new_item = np.array([interpolated_x, interpolated_y]).T
        newls.append(new_item)
    return newls

def new_df(trips, nb_points, mask, newlist): 
    """
    Create a new dataframe with the new interpolated version of trajectories (50 coordinates) and saves it outside gitHub

    Args:
        trips (pd.DataFrame): Dataframe with a fixed row length containing: trajectory id and trajectories 
        nb_points (int): Number of rows selected on dataframe trips
        mask (np.array): matrix indicating whether trajectories have more than one coordinate or not 
        newlist (list): interpolated trajectories 

    """
    ids = trips.iloc[0:nb_points][mask]["TRIP_ID"].to_numpy()
    new_list = np.array(newlist)
    new_df = []
    for i in range(len(newlist)): 
        new_df.append(str(new_list[i].tolist()))
    pd.DataFrame(np.array(new_df), index=ids, columns={'POLYLINE'}).to_csv('../taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015/interpolation'+str(nb_points)+'.csv')
    return print('ok')
