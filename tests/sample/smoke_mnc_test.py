# -*- coding: utf-8 -*-
import numpy as np
import math
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pytest

from deepctr_torch.sample.smoke_mnc import SMOTEMNC
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def test_smokeenc():
    # if version.parse(torch.__version__) >= version.parse("1.1.0") and len(dnn_hidden_units)==0:#todo check version
    #     return
    sen = SMOTEMNC(categorical_features=[1])
    assert 1 == 1


data = np.array(
    [
        [1.0, "1", "1"],
        [1.1, "1", "1"],
        [1.2, "1", "1"],
        [1.3, "1", "1"],
        [1.4, "1", "1"],
        [1.5, "1", "1"],
        [1.6, "1", "1"],
        [1.7, "1", "1"],
        [1.8, "1", "1"],
        [1.9, "1", "1"],
        [1.11, "1", "1"],
        [1.12, "1", "1"],
        [1.13, "1", "1"],
        [1.14, "1", "1"],
        [1.15, "1", "1"],
        [1.16, "1", "1"],
        [2.0, "2", "2"],
        [2.1, "2", "2"],
        [2.2, "2", "2"],
        [2.3, "2", "2"],
        [2.4, "2", "2"],
        [2.5, "2", "2"],
        [2.6, "2", "2"],
        [3.1, "3", "2"],
        [3.2, "3", "2"],
        [3.3, "3", "2"],
        [3.4, "3", "2"],
        [3.5, "3", "2"],
    ],
    dtype="object",
)

np.random.seed(2023)


def prepare_data() -> pd.DataFrame:
    _col_num = 4
    _col1 = np.random.random(
        len(data[:, 0])*_col_num).reshape(len(data[:, 0]), _col_num)
    data_new = np.append(data, _col1, axis=1)
    pd_data = pd.DataFrame(data_new, index=range(
        data.shape[0]), columns=['n0', 'c3', 'target', 'n1', 'n2', 'n3', 'n4'])
    return pd_data


def test_smote_enc_3bin():
    # Non-regression test for #662
    # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/662

    # pd_data = pd.DataFrame(data, index=range(
    #     data.shape[0]), columns=['n0', 'c3', 'target'])

    pd_data = prepare_data()
    # print(pd_data)
    cm = pd_data[['n0', 'n1', 'n2', 'n3', 'n4']].astype(float).cov().to_numpy()
    np.set_printoptions(precision=4, suppress=True)
    ivt_cm = np.linalg.inv(cm)
    print("\nInverse covar matrix: ")
    print(ivt_cm)

    pd_data['c3'] = LabelEncoder().fit_transform(pd_data['c3'])
    pd_data['target'] = LabelEncoder().fit_transform(pd_data['target'])

    pd_data.index = pd.RangeIndex(len(pd_data.index))

    scaler = MinMaxScaler(feature_range=(0, 1))
    pd_data['n0'] = scaler.fit_transform(
        np.array(pd_data['n0']).reshape(-1, 1))

    sc = StandardScaler()
    pd_data['c3'] = sc.fit_transform(
        np.array(pd_data['c3']).reshape(-1, 1))

    pd_label = pd_data[['target']]
    pd_data.drop('target', axis=1, inplace=True)
    print(pd_data['c3'].value_counts())
    # print(pd_data.head(4))
    smote = SMOTEMNC(categorical_features=[
        1], target_column='target', sampling_strategy=1, cov=ivt_cm)
    X_res, y_res = smote.fit_resample(pd_data, pd_label)

    print(X_res['c3'].value_counts())
    # assert len(X_res['c3'].unique()) == len(pd_data['c3'].unique())


# def test_smote_nc_3bin():
#     print("---")
#     # Non-regression test for #662
#     # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/662

#     pd_data = pd.DataFrame(data, index=range(
#         data.shape[0]), columns=['n0', 'c3', 'target'])

#     pd_data['c3'] = LabelEncoder().fit_transform(pd_data['c3'])
#     pd_data['target'] = LabelEncoder().fit_transform(pd_data['target'])

#     pd_data.index = pd.RangeIndex(len(pd_data.index))

#     pd_label = pd_data[['target']]
#     pd_data.drop('target', axis=1, inplace=True)
#     print(pd_data['c3'].value_counts())
#     smote = SMOTENC(categorical_features=[
#         1], sampling_strategy=1)
#     X_res, y_res = smote.fit_resample(pd_data, pd_label)

#     print(X_res['c3'].value_counts())
#     assert len(X_res['c3'].unique()) == len(pd_data['c3'].unique())


# def test_smote_enc_2bin():
#     print("---")
#     # Non-regression test for #662
#     # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/662
#     data = np.array(
#         [
#             [1.0, "1", "1"],
#             [1.1, "1", "1"],
#             [1.2, "1", "1"],
#             [1.3, "1", "1"],
#             [1.4, "1", "1"],
#             [1.5, "1", "1"],
#             [1.6, "1", "1"],
#             [1.7, "1", "1"],
#             [1.8, "1", "1"],
#             [1.9, "1", "1"],
#             [1.11, "1", "1"],
#             [2.0, "2", "2"],
#             [2.1, "2", "2"],
#             [2.2, "2", "2"],
#             [2.3, "2", "2"],
#             [2.4, "2", "2"],
#             [2.5, "2", "2"],
#         ],
#         dtype="object",
#     )

#     pd_data = pd.DataFrame(data, index=range(
#         data.shape[0]), columns=['n0', 'c3', 'target'])

#     pd_data['c3'] = LabelEncoder().fit_transform(pd_data['c3'])
#     pd_data['target'] = LabelEncoder().fit_transform(pd_data['target'])

#     pd_data.index = pd.RangeIndex(len(pd_data.index))

#     pd_label = pd_data[['target']]
#     pd_data.drop('target', axis=1, inplace=True)

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     pd_data['n0'] = scaler.fit_transform(
#         np.array(pd_data['n0']).reshape(-1, 1))

#     sc = StandardScaler()
#     pd_data['c3'] = sc.fit_transform(
#         np.array(pd_data['c3']).reshape(-1, 1))

#     print(pd_data['c3'].value_counts())
#     smote = SMOTEENC(categorical_features=[
#                      1], target_column='target', sampling_strategy=1)
#     X_res, y_res = smote.fit_resample(pd_data, pd_label)

#     print(X_res['c3'].unique())
#     assert len(X_res['c3'].unique()) == len(pd_data['c3'].unique())
