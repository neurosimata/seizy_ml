# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
##### ------------------------------------------------------------------- #####


def feature_selection_and_ranking(x, y_true, feature_labels):
    """
    This function performs feature selection by eliminating highly correlated features and then ranks 
    the selected features using ANOVA F-value and mutual information.
    
    Parameters:
    x (numpy array): Array containing feature values
    y_true (numpy array): Array containing true labels
    feature_labels (list): List of feature labels
    
    Returns:
    None. A csv file named 'feature_metrics.csv' is saved in the current directory.
    """

    # Select features that are not highly correlated before proceeding with ranking
    r_threshold = 0.99
    corr_matrix = np.corrcoef(x.T)
    corr_matrix = pd.DataFrame(corr_matrix, index=feature_labels, columns=feature_labels)
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_pairs = corr_matrix[corr_matrix > r_threshold].stack().index.tolist()
    
    # Select features from high correlated features
    corr_feature_df = pd.DataFrame(correlated_pairs)
    high_corr_features = set(np.unique(corr_feature_df[corr_feature_df.columns].values))
    for feature1, feature2 in correlated_pairs:
        corr_feature_df = corr_feature_df[corr_feature_df.iloc[:,0] != feature2]
    
    selected_high_corr_features = set(corr_feature_df.iloc[:,0].values)
    drop_features = high_corr_features.difference(selected_high_corr_features)
    selected_features = set(corr_matrix.index).difference(drop_features)
    
    # put them back in feature original order
    selected_feature_idx = np.zeros(len(feature_labels), dtype=bool)
    for feature in selected_features:
        selected_feature_idx[corr_matrix.index == feature] = True
    selected_features = feature_labels[selected_feature_idx]
    
    # Get f-values based on ANOVA
    f_vals_anova, _ = f_classif(x[:,selected_feature_idx], y_true)

    # Get MI based on mutual information
    mutual_info = mutual_info_classif(x[:, selected_feature_idx], y_true)
    
    # Map back to full feature space
    anova_ranks = np.empty(len(feature_labels), dtype=np.dtype('U100'))
    sorted_indices = (1/f_vals_anova).argsort()
    ranks = sorted_indices.argsort() + 1
    for feature, rank in zip(selected_features, ranks):
        anova_ranks[corr_matrix.index == feature] = str(rank)
 
    # Map back to full feature space
    mutual_info_ranks = np.empty(len(feature_labels), dtype=np.dtype('U100'))
    sorted_indices = (1/mutual_info).argsort()
    ranks = sorted_indices.argsort() + 1
    for feature, rank in zip(selected_features, ranks):
        mutual_info_ranks[corr_matrix.index == feature] = str(rank)
    
    # Get metrics dataframe and store
    feature_metrics = pd.DataFrame(data=[
                                ['anova_ranks'] + list(anova_ranks),
                                ['mutual_info_ranks'] + list(mutual_info_ranks)],
                                columns=['metrics'] + list(feature_labels))
    return feature_metrics


def get_feature_space(feature_ranks, feature_size=[33,66],
                  export_name='feature_space.csv'):
    """
    Create feature space from anova and mutual info ranks dataframe.

    Parameters
    ----------
    feature_ranks : pandas df, containing mnetrics
    feature_size : list, Percent of features to select from ANOVA and MI ranks.

    Returns
    -------
    feature_space : pandas df,
    dft : pandas df,

    """

    # reformat dataframe to get features
    dft = dft = feature_ranks.drop('metrics', axis=1)
    dft = dft.T.reset_index()
    dft.columns = ['features'] + list(feature_ranks['metrics'])
    df = feature_ranks.drop('metrics', axis=1)
    df = df.apply(pd.to_numeric)
    
    # get rank columns and set empty ranks(discarded because of high correlation) to max length
    rank_cols = [col for col in dft.columns if 'ranks' in col]
    dft.loc[dft[rank_cols[0]] == '', rank_cols]  = str(len(dft))
    dft[rank_cols] = dft[rank_cols].astype(int)
    
    # get index
    bool_array = np.zeros((len(rank_cols)*len(feature_size)+1, len(df.columns)))
    max_rank = int(np.nanmax(df.values))
    
    # add all non correlated features
    bool_array[0,:] = dft[rank_cols[0]]<=max_rank
    cntr = 1
    for rank_col in rank_cols:
        for feature_percent in feature_size:   
            feature_len = int(feature_percent/100*len(feature_ranks.columns))
            idx = dft.nsmallest(feature_len, columns=rank_col).index.values
            bool_array[cntr, idx] = 1
            cntr+=1
            
    # merge dfs
    feature_space = pd.DataFrame(columns=df.columns, data=bool_array)
    return feature_space



    
    


