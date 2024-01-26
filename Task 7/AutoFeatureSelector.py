import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Model imports
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# By: Syed Hamza Ali


# Cor Selector Function
def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


# Chi Squared SFunction
def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    return chi_support, chi_feature


# RFE Selector Function
def rfe_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(),
                       n_features_to_select=num_feats,
                       step=10,
                       verbose=1)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    return rfe_support, rfe_feature


# Embedded Lasso Reguarization Selector Function
def embedded_log_reg_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    # Note: with Ibgfs, l1 penalty would give error, therefore used linlinear
    embedded_lr_selector = SelectFromModel(LogisticRegression(solver='liblinear', penalty="l1", random_state=33),
                                           max_features=num_feats)
    embedded_lr_selector.fit(X_norm, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature


# Tree Based (Random Forest)
def embedded_rf_selector(X, y, num_feats):
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature


# Tree Based (Light GBM)
def embedded_lgbm_selector(X, y, num_feats):
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3,
                          min_split_gain=0.01, min_child_weight=40)
    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature


# Preprocessing Function
def preprocess_dataset(dataset_path):
    player_df = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
               'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    player_df = player_df[numcols + catcols]
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf, columns=features)

    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']

    # no of maximum features we need to select
    num_feats = 30

    return X, y, num_feats


# AutoFeatureSelector Function
def AutoFeatureSelector(dataset_path, methods=[]):

    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)

    selector_choices = {}

    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    feature_name = list(X.columns)

    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
        selector_choices['Pearson'] = cor_support
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
        selector_choices['chi-square'] = chi_support
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
        selector_choices['rfe'] = rfe_support
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        selector_choices['log-reg'] = embedded_lr_support
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        selector_choices['rf'] = embedded_rf_support
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        selector_choices['lgbm'] = embedded_lgbm_support

    pd.set_option('display.max_rows', None)
    feature_selection_df = pd.DataFrame(
        {'Feature': feature_name, **selector_choices})

    feature_selection_df['Total'] = np.sum(feature_selection_df.iloc[:, 1:], axis=1)

    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)

    best_features = feature_selection_df.head(num_feats)

    return best_features


if __name__ == '__main__':

    input_str = input("For the dataset fif19.csv, Enter the required methods (space separated)(ex: pearson chi-square rfe log-reg rf lgbm)(any order or amount):  ")
    input_array = input_str.split()

    best_features = AutoFeatureSelector(dataset_path="data/fifa19.csv", methods=input_array)
    print('\n\nThe top 5 best features are: \n', best_features['Feature'].iloc[1:6].to_string(index=False),'\n\n')
    best_features['Feature'].iloc[1:6].to_csv('./output.txt', sep='\t', index=False)
    print('Thes results have been written and saved in output.txt.')
