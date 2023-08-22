# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
##### ------------------------------------------------------------------- #####


def feature_selection_and_ranking(x, y_true, feature_labels, r_threshold=0.99):
    """
    This function performs feature selection by eliminating highly correlated features and then ranks 
    the selected features using ANOVA F-value and mutual information.
    
    Parameters:
    x (numpy array): Array containing feature values
    y_true (numpy array): Array containing true labels
    feature_labels (list): List of feature labels.
    r_threshold (float): Remove highly correlated features above the value.
    
    Returns:
    None. A csv file named 'feature_metrics.csv' is saved in the current directory.
    """

    # Select features that are not highly correlated before proceeding with ranking
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
    list(corr_matrix.index)
    
    selected_feature_idx = np.zeros(len(feature_labels), dtype=bool)
    for feature in selected_features:
        selected_feature_idx[corr_matrix.index == feature] = True
    
    # Get f-values based on ANOVA
    f_vals_anova, _ = f_classif(x[:,selected_feature_idx], y_true)
    
    # Get MI based on mutual information
    mutual_info = mutual_info_classif(x[:, selected_feature_idx], y_true)
    
    # Map back to full feature space
    anova_ranks = np.empty(len(feature_labels), dtype=np.dtype('U100'))
    sorted_indices =  (1/f_vals_anova).argsort()
    ranks = sorted_indices.argsort() + 1
    for feature, rank in zip(selected_features, ranks):
        anova_ranks[corr_matrix.index == feature] = str(rank)
        
    # Map back to full feature space
    mutual_info_ranks = np.empty(len(feature_labels), dtype=np.dtype('U100'))
    sorted_indices =  (1/mutual_info).argsort()
    ranks = sorted_indices.argsort() + 1
    for feature, rank in zip(selected_features, ranks):
        mutual_info_ranks[corr_matrix.index == feature] = str(rank)
    
    # Get metrics dataframe and store
    feature_metrics = pd.DataFrame(data=[
                                ['anova_ranks'] + list(anova_ranks),
                                ['mutual_info_ranks'] + list(mutual_info_ranks)],
                                columns=['metrics'] + list(feature_labels))
    
    feature_metrics.to_csv('feature_metrics.csv', index=False)


def feature_space(feature_metrics, feature_size=[4,8],
                  export_name='feature_space.csv', export=True):
    """
    Create feature space from anova and mutual info ranks dataframe.

    Parameters
    ----------
    feature_metrics : pandas df, containing mnetrics
    feature_size : list, The default is [4,8].
    export_name : str, The default is 'feature_space.csv'.
    export : bool, The default is True.

    Returns
    -------
    feature_space : pandas df,
    dft : pandas df,

    """

    # reformat dataframe to get features
    dft = dft = feature_metrics.drop('metrics', axis=1)
    dft = dft.T.reset_index()
    dft.columns = ['features'] + list(feature_metrics['metrics'])
    df = feature_metrics.drop('metrics', axis=1)
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
        for feature_len in feature_size:
            idx = dft.nsmallest(feature_len, columns=rank_col).index.values
            bool_array[cntr, idx] = 1
            cntr+=1
            
    # merge dfs
    feature_space = pd.DataFrame(columns=df.columns, data=bool_array)
    if export:
        feature_space.to_csv(export_name, index=False)
    return feature_space, ranks


if __name__ == '__main__':
    
    data = np.random.rand(51,500,3)
    
    # get feature metrics
    feature_metrics = feature_selection_and_ranking(x, y_true, feature_labels)
    
    # create feature space df for training
    feature_space, ranks = feature_space(feature_metrics, feature_size=[4,8],
                      export_name='feature_space.csv', export=True)
    
    


