import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.utils.logger import logger

pd.set_option('display.width', 150)

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
    
    # Visualize the variance explanations
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.xticks(rotation=45, ha='right')
    plt.bar(x=main_features, height=explained_variance)
    plt.title(f'PCA components variance explained, named by the most contributing feature')

    # Visualize the feature contributions
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.set(font_scale=0.5)
    sns.heatmap(feature_conts_df, annot=False, cmap=sns.color_palette("Greens", 256), linewidths=0.5, linecolor='gray')
    plt.title(f'The actual feature contributions to the PCA components')
    
# Extract information about the PCA run
def get_pca_run_stats(pca, input_features):
    pca_feature_names = pca.get_feature_names_out(input_features)
    logger.info(f'The PCA feature name out:\n{pca_feature_names}')

    # Prepare the components relations with features
    raw_relation_df = pd.DataFrame(pca.components_, columns=input_features, index = pca_feature_names)
    # Take the absolute values as the sign does not matter
    raw_relation_df = raw_relation_df.abs()

    # For each PCA component get a descending list of feature importances:
    # feature_conts = raw_relation_df.apply(lambda row: list(reversed(sorted(zip(row.values, input_features)))), axis=1)
    # feature_conts_df = pd.DataFrame({'feature_contributions': feature_conts }, index=pca_feature_names)
    # feature_map = {f'pca{idx}' : feature_conts_df.loc[f'pca{idx}']['feature_contributions'][0][1] for idx in range(len(pca_feature_names))}

    # Get the most features contributing most to the PCA components. This is a 
    # simpler version than the above if we only need the most contributing feature
    feature_conts_df = raw_relation_df.idxmax(axis=1)
    feature_map = {f'pca{idx}' : feature_conts_df.loc[f'pca{idx}'] for idx in range(len(pca_feature_names))}

    main_features = [feature_map[pca_name] for pca_name in pca_feature_names]
    explained_variance = pca.explained_variance_ratio_

    return main_features, raw_relation_df, explained_variance