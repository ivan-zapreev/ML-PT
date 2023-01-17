import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.utils.logger import logger
from src.features.utils import plot_2d_feature_space
  
# Plot the 2D space projected clusters
def plot_2d_feature_space_clusters(X_proj, clusterer, palette=None, title=None):
    # First divise the colors
    color_palette = sns.color_palette('Paired', 100) if palette is None else palette[1:]
    cluster_colors = [color_palette[x] if x >= 0 else (0, 0, 0) for x in clusterer.labels_]
    
    # Compute the noize level
    cluster_labels, cluster_sizes = np.unique(clusterer.labels_, return_counts=True)
    noize_index = np.argwhere(cluster_labels == -1)[0][0]
    num_noize_samples = cluster_sizes[noize_index]
    noize = num_noize_samples*100/len(clusterer.labels_)
    logger.info(f'Found noize label index: {noize_index}, noize samples count: {num_noize_samples}, noize: {round(noize, 2)} %')
    
    # Plot the results
    if title is None:
        title = 'The feature clusters 2D projectsion'
    plot_2d_feature_space(X_proj, c=cluster_colors, title=f'{title}, noize level={round(noize, 2)}%')

# Visualize the Elbow/Knee value found when searching for DBSCAN eps
def visualize_elbow_eps_value(distances, x_pos, y_pos, title):
    fig, ax = plt.subplots(figsize=(15, 3))
    plt.plot(distances)
    ax.set_yscale('log')
    plt.plot([x_pos, x_pos], [-10, 40], 'k--', lw=1)
    plt.plot([-10, 60000], [y_pos, y_pos], 'k--', lw=1)
    plt.title(title + f', eps: {round(y_pos, 4)}')

def classify_attack_clusters(data, num_attack_classes=50):
    # Extract the cluster data
    cluster_labels = data['cluster_labels']
    cluster_sizes = data['cluster_sizes']
    
    # Remove the first element as it is the noize cluster then join
    # labels with sizes and make a size descending list of cluster labels
    cluster_data = list(zip(cluster_sizes, cluster_labels))[1:]
    cluster_data = list(reversed(sorted(cluster_data)))

    # Select the first classes, except the last num_attack_classes classes to be the good ones
    unkl_ids = [-1]
    good_ids = sorted([entry[1] for entry in cluster_data[:len(cluster_data) - num_attack_classes]])
    attack_ids = sorted([entry[1] for entry in cluster_data[- num_attack_classes:]])

    # Produce the colors mapping
    colors = ['black' if clust_id == -1 else 'green' if clust_id in good_ids else 'red' for clust_id in range(-1, len(cluster_labels)-1)]
    return unkl_ids, good_ids, attack_ids, colors

def plot_cluster_sizes(data, title, color):
    cluster_labels = data['cluster_labels']
    cluster_sizes = data['cluster_sizes']

    fig, ax = plt.subplots(figsize=(15, 3))
    _ = plt.bar(x = cluster_labels, height = cluster_sizes, color=color)
    plt.title(title + f', num clusters: {len(cluster_labels)}')
    ax.set_xlabel('Cluster id')
    ax.set_ylabel('Cluster size')

# Plot cluster sizes for specified S values
def plot_cluster_sizes_for_s(s_values, s_results, s_value):
    # Get the corresponding data
    data = s_results[s_values.index(s_value)]
    
    # Plot the cluster sizes
    plot_cluster_sizes(data, title='Cluster sizes for S: {s_value}')
    
    # Visualize the elbow curve
    visualize_elbow_eps_value(**data['elbow'], title=f'Elbow detection for S: {s_value}')
    
    return data