# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
##### ------------------------------------------------------------------- #####


def select_features(x, y_true, feature_labels, r_threshold, feature_size, nleast_correlated=0):
    """
     Selects features based on ANOVA F-values, while dropping highly correlated features 
     according to their ANOVA ranks, and optionally includes the least correlated features.
    
     Parameters:
     - x (array-like): The feature matrix with shape (n_samples, n_features).
     - y_true (array-like): The true class labels with shape (n_samples,).
     - feature_labels (array-like): The feature labels corresponding to columns from x.
     - r_threshold (float): Correlation threshold for feature selection. Features with correlation 
       higher than this threshold will be considered for dropping.
     - feature_size (list of int): List of sizes for the feature subsets to be selected.
     - nleast_correlated (int): Number of least correlated features to be included 
       in the final selection.
    
     Returns:
     - feature_dict (dict): A dictionary with keys corresponding to the different feature subsets.
       Each key has two variations:
         - 'best_{size}': Top features based solely on ANOVA ranks.
         - 'best_{size}_and_leastcorr': Top features combined with the least correlated features.
    """
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(x.T)
    corr_df = pd.DataFrame(corr_matrix, index=feature_labels, columns=feature_labels)

    # Perform ANOVA
    f_vals_anova, _ = f_classif(x, y_true)
    anova_ranks = pd.Series(f_vals_anova, index=feature_labels).rank(method='min', ascending=False)
    
    # Identify highly correlated feature pairs
    upper_triangle = np.triu(np.ones(corr_df.shape), k=1).astype(bool)
    high_corr_pairs = corr_df.where(upper_triangle).stack()
    high_corr_pairs = high_corr_pairs[high_corr_pairs > r_threshold].index.tolist()

    # Determine which feature to drop based on lower ANOVA rank
    to_drop = set()
    for feature1, feature2 in high_corr_pairs:
        if anova_ranks[feature1] > anova_ranks[feature2]:
            to_drop.add(feature1)
        else:
            to_drop.add(feature2)
    selected_features = [str(f) for f in feature_labels if f not in to_drop]
    
    # Use the original ANOVA ranks to sort the selected features
    selected_anova_ranks = anova_ranks[selected_features]

    # Find the n least correlated features to the top 5 features
    top_5_features = selected_anova_ranks.nsmallest(5).index.values
    corr_to_top_5 = corr_df.loc[:, top_5_features].abs().mean(axis=1)
    least_corr_features = corr_to_top_5.nsmallest(nleast_correlated).index.tolist()

    # Create the feature dictionary for the specified feature sizes
    feature_dict = {}
    for size in feature_size:
        top_features = selected_anova_ranks.nsmallest(size).index.tolist()
        feature_dict[f'best_{str(size)}'] = top_features
        
    if nleast_correlated > 0:
        for size in feature_size:
            top_features = selected_anova_ranks.nsmallest(size).index.tolist()
            feature_dict[f'best_{str(size)}_and_leastcorr'] =  top_features + [f for f in least_corr_features if f not in top_features]

    return feature_dict


    
    


