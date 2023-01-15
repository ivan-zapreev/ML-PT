import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.utils.logger import logger

# Create the featre space projection
def create_2d_projection(X):
    return TSNE(n_components=2, learning_rate='auto', init='pca', n_jobs=-1).fit_transform(X)

# Plot the multi dimensional space projction
def plot_2d_feature_space(X_proj, s=5, c='black'):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(*X_proj.T, s=s, c=c, linewidth=0, alpha=0.25)
    plt.title(f'The 2D feature projection space visualized')
    
# Plot the 2D space projected clusters
def plot_2d_feature_space_clusters(X_proj, clusterer):
    # First divise the colors
    color_palette = sns.color_palette('Paired', 100)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0, 0, 0)
                      for x in clusterer.labels_]
    
    # Plot the results
    plot_2d_feature_space(X_proj, c=cluster_colors)

    
# Define the plotting function
def plot_variance_explained(extractor):
    feature_names, var_values, raw_relation_df = extractor.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.xticks(rotation=45, ha='right')
    plt.bar(x=feature_names, height=var_values)
    plt.title(f'The variance explained by main PCA components represented by their most contributing features')