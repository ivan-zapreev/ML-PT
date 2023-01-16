import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.utils.logger import logger

# Create the featre space projection
def create_2d_projection(X):
    return TSNE(n_components=2, learning_rate='auto', init='pca', n_jobs=-1, random_state=0).fit_transform(X)

# Plot the multi dimensional space projction
def plot_2d_feature_space(X_proj, s=5, c='black', title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(*X_proj.T, s=s, c=c, linewidth=0, alpha=0.25)
    plt.title(f'The 2D feature projection space visualized' if title is None else title)
    
# Define the plotting function
def plot_variance_explained(extractor):
    main_features, feature_conts_df, explained_variance = extractor.get_feature_names_out()
    
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.xticks(rotation=45, ha='right')
    plt.bar(x=main_features, height=explained_variance)
    plt.title(f'The variance explained by main PCA components represented by their most contributing features')
    
# Extract information about the PCA run
def get_pca_run_stats(pca, input_features):
    pca_feature_names = pca.get_feature_names_out(input_features)
    logger.info(f'The PCA feature name out:\n{pca_feature_names}')

    # Prepare the components relations with features
    raw_relation_df = pd.DataFrame(pca.components_, columns=input_features, index = pca_feature_names)
    # Take the absolute values as the sign does not matter
    raw_relation_df = raw_relation_df.abs()

    # TODO: Instead of dumping raw_relation_df and relation_df, dump the top features contributing to the PCA components with values

    # Get the most features contributing most to the PCA components
    feature_conts_df = raw_relation_df.idxmax(axis=1)

    feature_map = {f'pca{idx}' : feature_conts_df.loc[f'pca{idx}'] for idx in range(len(pca_feature_names))}
    main_features = [feature_map[pca_name] for pca_name in pca_feature_names]
    explained_variance = pca.explained_variance_ratio_

    return main_features, feature_conts_df, explained_variance