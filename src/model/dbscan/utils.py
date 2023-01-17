import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.utils.logger import logger
from src.features.utils import plot_2d_feature_space
  
# Plot the 2D space projected clusters
def plot_2d_feature_space_clusters(X_proj, clusterer):
    # First divise the colors
    color_palette = sns.color_palette('Paired', 100)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0, 0, 0)
                      for x in clusterer.labels_]
    
    # Compute the noize level
    cluster_labels, cluster_sizes = np.unique(clusterer.labels_, return_counts=True)
    noize_index = np.argwhere(cluster_labels == -1)[0][0]
    num_noize_samples = cluster_sizes[noize_index]
    noize = num_noize_samples*100/len(clusterer.labels_)
    logger.info(f'Found noize label index: {noize_index}, noize samples count: {num_noize_samples}, noize: {round(noize, 2)} %')
    
    # Plot the results
    plot_2d_feature_space(X_proj, c=cluster_colors, title=f'The feature clusters 2D projectsion, noize level={round(noize, 2)}%')

# Visualize the Elbow/Knee value found when searching for DBSCAN eps
def visualize_elbow_eps_value(distances, x_pos, y_pos, title):
    fig, ax = plt.subplots(figsize=(15, 3))
    plt.plot(distances)
    ax.set_yscale('log')
    plt.plot([x_pos, x_pos], [-10, 40], 'k--', lw=1)
    plt.plot([-10, 60000], [y_pos, y_pos], 'k--', lw=1)
    plt.title(title + f', eps: {round(y_pos, 4)}')

# Plot cluster sizes for specified S values
def plot_cluster_sizes_for_s(s_values, s_results, s_value):
    fig, ax = plt.subplots(figsize=(15, 3))
    data = s_results[s_values.index(s_value)]
    cluster_labels = data['cluster_labels']
    cluster_sizes = data['cluster_sizes']
    _ = plt.bar(x = cluster_labels, height = cluster_sizes)
    plt.title(f'Cluster sizes for S: {s_value}, num clusters: {len(cluster_labels)}')
    ax.set_xlabel('Cluster id')
    ax.set_ylabel('Cluster size')
    
    # Visualize the elbow curve
    visualize_elbow_eps_value(**data['elbow'], title=f'Elbow detection for S: {s_value}')
    
    return data