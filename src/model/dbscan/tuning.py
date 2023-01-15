import kneed

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.utils.logger import logger

def compute_elbouw_eps_value(X, min_samples, s_value):
    # Devise the optimal value of eps for DBSCAN
    nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
    nearest_neighbors.fit(X)
    distances, _ = nearest_neighbors.kneighbors(X)

    # Extract and sort
    distances = np.sort(distances, axis=0)[:, 1]
    logger.debug(f'Resulting distance shape for min_samples: {min_samples} is: {distances.shape}')

    # Detect the knee
    kneedle = kneed.KneeLocator(range(len(distances)), distances, S=s_value, curve='convex', direction='increasing')
    x_pos = kneedle.elbow
    y_pos = distances[kneedle.elbow]
    logger.debug(f'The detected elbouw point min_samples: {min_samples} and S: {s_value} is: ({x_pos}, {round(y_pos, 2)})')
    
    return distances, x_pos, y_pos

def visualize_elbow_eps_value(distances, x_pos, y_pos, title):
    fig, ax = plt.subplots(figsize=(15, 3))
    plt.plot(distances)
    ax.set_yscale('log')
    plt.plot([x_pos, x_pos], [-10, 40], 'k--', lw=1)
    plt.plot([-10, 60000], [y_pos, y_pos], 'k--', lw=1)
    plt.title(title + f', eps: {round(y_pos, 2)}')

def compute_dbscan_clusters(X, eps_value, min_samples):
    # Apply DBSCAN for clustering of the provided data
    logger.debug(f'Start fitting DBSCAN model')
    clusterer = DBSCAN(eps=eps_value, min_samples=min_samples, n_jobs=-1).fit(X)
    
    # List the labels to understand how many clusters we have
    cluster_labels, cluster_sizes = np.unique(clusterer.labels_, return_counts=True)
    num_labels = len(cluster_labels)
    logger.debug(f'There are {num_labels} distinct DBSCAN cluster labels found')
    
    return clusterer, cluster_labels, cluster_sizes

def fit_dbscan_clusters(X, min_samples, s_value):
    # First, find the optimal number value for epsilon
    distances, x_pos, y_pos = compute_elbouw_eps_value(X, min_samples, s_value)
    eps_value = y_pos
    
    # Second, cluster the data and get the number of clusters
    clusterer, cluster_labels, cluster_sizes = compute_dbscan_clusters(X, eps_value, min_samples)
    
    # Return all the relevant data
    return {
              'elbow' : {
                          'distances' : distances,
                          'x_pos' : x_pos,
                          'y_pos' : y_pos
                        },
             'eps_value' : eps_value,
             'clusterer' : clusterer,
             'cluster_labels' : cluster_labels,
             'cluster_sizes' : cluster_sizes
            }