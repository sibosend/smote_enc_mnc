# Our New Proposed SMOTE Method
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.utils import check_array, sparsefuncs_fast, check_X_y, check_random_state
from sklearn.utils import _safe_indexing as safe_indexing
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from numbers import Integral
from sklearn.svm import SVC
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE
import os
# import missingpy as missingpy
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import pickle
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors._base import KNeighborsMixin
from imblearn.exceptions import raise_isinstance_error
from imblearn.utils import check_target_type


class SMOTEENC(SMOTE):

    def __init__(self,
                 categorical_features,
                 *,
                 sampling_strategy="auto",
                 random_state=None,
                 k_neighbors=5,
                 n_jobs=None,
                 target_column='target'):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.target_value = 1

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X, y, reset=True, dtype=None, accept_sparse=["csr", "csc"]
        )
        # print("binarize_y", binarize_y)
        return X, y, binarize_y

    def chk_neighbors(self, nn_object, additional_neighbor):
        if isinstance(nn_object, Integral):
            return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
        elif isinstance(nn_object, KNeighborsMixin):
            return clone(nn_object)
        else:
            raise_isinstance_error(self, [int, KNeighborsMixin], nn_object)

    def generate_samples(self, X, nn_data, nn_num, rows, cols, steps, continuous_features_,):
        rng = check_random_state(42)

        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs

        X_new = (X_new.tolil() if sparse.issparse(X_new) else X_new)
        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = (nn_data.toarray() if sparse.issparse(nn_data) else nn_data)

        all_neighbors = nn_data[nn_num[rows]]

        for idx in range(continuous_features_.size, X.shape[1]):

            mode = stats.mode(all_neighbors[:, :, idx], axis=1)[0]

            X_new[:, idx] = np.ravel(mode)
        return X_new

    def make_samples(self, X, y_dtype, y_type, nn_data, nn_num, n_samples, continuous_features_, step_size=1.0):
        random_state = check_random_state(42)
        samples_indices = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples)
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self.generate_samples(
            X, nn_data, nn_num, rows, cols, steps, continuous_features_)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)

        return X_new, y_new

    def cat_corr_pandas(self, X, target_df):
        # X has categorical columns
        categorical_columns = list(X.columns)
        X = pd.concat([X, target_df], axis=1)

        # filter X for target value
        is_target = X.loc[:, self.target_column] == self.target_value
        X_filtered = X.loc[is_target, :]

        X_filtered.drop(self.target_column, axis=1, inplace=True)

        # get columns in X
        nrows = len(X)
        encoded_dict_list = []
        nan_dict = dict({})
        c = 0
        imb_ratio = len(X_filtered)/len(X)
        # print("imb_ratio", imb_ratio)
        OE_dict = {}

        for column in categorical_columns:
            for level in list(X.loc[:, column].unique()):
                # filter rows where level is present
                row_level_filter = X.loc[:, column] == level
                rows_in_level = len(X.loc[row_level_filter, :])
                # number of rows in level where target is 1
                O = len(X.loc[is_target & row_level_filter, :])
                E = rows_in_level * imb_ratio
                # Encoded value = chi, i.e. (observed - expected)/expected
                ENC = (O - E) / E
                OE_dict[level] = ENC

            encoded_dict_list.append(OE_dict)

            X.loc[:, column] = X[column].map(OE_dict)
            nan_idx_array = np.ravel(np.argwhere(
                np.array(np.isnan(X.loc[:, column]))))
            if len(nan_idx_array) > 0:
                nan_dict[c] = nan_idx_array
            c = c + 1
            X.loc[:, column].fillna(-1, inplace=True)

        X.drop(self.target_column, axis=1, inplace=True)
        return X, encoded_dict_list, nan_dict

    def _fit_resample(self, X, y):
        X = pd.DataFrame(X)
        y = pd.DataFrame(y, columns=[self.target_column])
        X_cat_encoded, encoded_dict_list, nan_dict = self.cat_corr_pandas(X.iloc[:, np.asarray(
            self.categorical_features)], y)
#         X_cat_encoded = np.ravel(np.array(X_cat_encoded))
        X_cat_encoded = np.array(X_cat_encoded)
        y = np.ravel(y)
        X = np.array(X)

        # print("Counter(y)", Counter(y))
        # unique, counts = np.unique(y, return_counts=True)
        # target_stats = dict(zip(unique, counts))
        # print("target_stats", target_stats)
        # n_sample_majority = max(target_stats.values())
        # class_majority = max(target_stats, key=target_stats.get)
        # sampling_strategy = {key: n_sample_majority - value for (
        #     key, value) in target_stats.items() if key != class_majority}

        # print("sampling_strategy", sampling_strategy)
        # print("self.sampling_strategy_", self.sampling_strategy_)
        # print("self.sampling_strategy_.items()",
        #       self.sampling_strategy_.items())
        n_features_ = X.shape[1]
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == 'bool':
            categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any([cat not in np.arange(n_features_) for cat in categorical_features]):
                raise ValueError('Some of the categorical indices are out of range. Indices'
                                 ' should be between 0 and {}'.format(n_features_))
            categorical_features_ = categorical_features

        continuous_features_ = np.setdiff1d(
            np.arange(n_features_), categorical_features_)

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=['csr', 'csc'])
        X_minority = safe_indexing(
            X_continuous, np.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == 'csr':
                _, var = sparsefuncs_fast.csr_mean_variance_axis0(X_minority)
            else:
                _, var = sparsefuncs_fast.csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, categorical_features_]
        X_copy = np.hstack((X_continuous, X_categorical))

        X_cat_encoded = X_cat_encoded * median_std_

        X_encoded = np.hstack((X_continuous, X_cat_encoded))
        X_resampled = X_encoded.copy()
        y_resampled = y.copy()
        # print("self.sampling_strategy_", self.sampling_strategy_)
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X_encoded, target_class_indices)
            nn_k_ = self.chk_neighbors(self.k_neighbors, 1)  # (5, 1)?
            nn_k_.fit(X_class)

            nns = nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self.make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, continuous_features_, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
                sparse_func = 'tocsc' if X.format == 'csc' else 'tocsr'
                X_resampled = getattr(X_resampled, sparse_func)()
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        X_resampled_copy = X_resampled.copy()
        i = 0
        for col in range(continuous_features_.size, X.shape[1]):
            encoded_dict = encoded_dict_list[i]
            i = i + 1
            for key, value in encoded_dict.items():
                X_resampled_copy[:, col] = np.where(np.round(X_resampled_copy[:, col], 4) == np.round(
                    value * median_std_, 4), key, X_resampled_copy[:, col])

        for key, value in nan_dict.items():
            for item in value:
                X_resampled_copy[item, continuous_features_.size +
                                 key] = X_copy[item, continuous_features_.size + key]

        X_resampled = X_resampled_copy
        indices_reordered = np.argsort(
            np.hstack((continuous_features_, categorical_features_)))
        if sparse.issparse(X_resampled):
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]
        return X_resampled, y_resampled


'''
class SMOTEENC_origin():

    def __init__(self, categorical_features, target_column):
        self.categorical_features = categorical_features
        self.target_column = target_column

    def chk_neighbors(self, nn_object, additional_neighbor):
        if isinstance(nn_object, Integral):
            return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
        elif isinstance(nn_object, KNeighborsMixin):
            return clone(nn_object)
        else:
            raise_isinstance_error(self, [int, KNeighborsMixin], nn_object)

    def generate_samples(self, X, nn_data, nn_num, rows, cols, steps, continuous_features_,):
        rng = check_random_state(42)

        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs

        X_new = (X_new.tolil() if sparse.issparse(X_new) else X_new)
        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = (nn_data.toarray() if sparse.issparse(nn_data) else nn_data)

        all_neighbors = nn_data[nn_num[rows]]

        for idx in range(continuous_features_.size, X.shape[1]):

            mode = stats.mode(all_neighbors[:, :, idx], axis=1)[0]
            X_new[:, idx] = np.ravel(mode)
        return X_new

    def make_samples(self, X, y_dtype, y_type, nn_data, nn_num, n_samples, continuous_features_, step_size=1.0):
        random_state = check_random_state(42)
        samples_indices = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples)
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self.generate_samples(
            X, nn_data, nn_num, rows, cols, steps, continuous_features_)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)

        return X_new, y_new

    def cat_corr_pandas(self, X, target_df, target_column, target_value):
        # X has categorical columns
        categorical_columns = list(X.columns)
        X = pd.concat([X, target_df], axis=1)

        # filter X for target value
        is_target = X.loc[:, target_column] == target_value
        X_filtered = X.loc[is_target, :]

        X_filtered.drop(target_column, axis=1, inplace=True)

        # get columns in X
        nrows = len(X)
        encoded_dict_list = []
        nan_dict = dict({})
        c = 0
        imb_ratio = len(X_filtered)/len(X)
        # print("imb_ratio", imb_ratio)
        OE_dict = {}

        for column in categorical_columns:
            for level in list(X.loc[:, column].unique()):
                # filter rows where level is present
                row_level_filter = X.loc[:, column] == level
                rows_in_level = len(X.loc[row_level_filter, :])
                # number of rows in level where target is 1
                O = len(X.loc[is_target & row_level_filter, :])
                E = rows_in_level * imb_ratio
                # Encoded value = chi, i.e. (observed - expected)/expected
                ENC = (O - E) / E
                OE_dict[level] = ENC

            encoded_dict_list.append(OE_dict)

            X.loc[:, column] = X[column].map(OE_dict)
            nan_idx_array = np.ravel(np.argwhere(
                np.array(np.isnan(X.loc[:, column]))))
            if len(nan_idx_array) > 0:
                nan_dict[c] = nan_idx_array
            c = c + 1
            X.loc[:, column].fillna(-1, inplace=True)

        X.drop(target_column, axis=1, inplace=True)
        return X, encoded_dict_list, nan_dict

    def fit_resample(self, X, y):
        # print(X, type(X))
        X_cat_encoded, encoded_dict_list, nan_dict = self.cat_corr_pandas(X.iloc[:, np.asarray(
            self.categorical_features)], y, target_column=self.target_column, target_value=1)
#         X_cat_encoded = np.ravel(np.array(X_cat_encoded))
        X_cat_encoded = np.array(X_cat_encoded)
        y = np.ravel(y)
        X = np.array(X)

        # print("Counter(y)", Counter(y))
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        # print("target_stats", target_stats)
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {key: n_sample_majority - value for (
            key, value) in target_stats.items() if key != class_majority}

        # print("sampling_strategy", sampling_strategy)

        n_features_ = X.shape[1]
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == 'bool':
            categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any([cat not in np.arange(n_features_) for cat in categorical_features]):
                raise ValueError('Some of the categorical indices are out of range. Indices'
                                 ' should be between 0 and {}'.format(n_features_))
            categorical_features_ = categorical_features

        continuous_features_ = np.setdiff1d(
            np.arange(n_features_), categorical_features_)

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=['csr', 'csc'])
        X_minority = safe_indexing(
            X_continuous, np.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == 'csr':
                _, var = sparsefuncs_fast.csr_mean_variance_axis0(X_minority)
            else:
                _, var = sparsefuncs_fast.csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, categorical_features_]
        X_copy = np.hstack((X_continuous, X_categorical))

        X_cat_encoded = X_cat_encoded * median_std_

        X_encoded = np.hstack((X_continuous, X_cat_encoded))
        X_resampled = X_encoded.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in sampling_strategy.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X_encoded, target_class_indices)
            nn_k_ = self.chk_neighbors(5, 1)  # (5, 1)?
            nn_k_.fit(X_class)

            nns = nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self.make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, continuous_features_, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
                sparse_func = 'tocsc' if X.format == 'csc' else 'tocsr'
                X_resampled = getattr(X_resampled, sparse_func)()
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        X_resampled_copy = X_resampled.copy()
        i = 0
        for col in range(continuous_features_.size, X.shape[1]):
            encoded_dict = encoded_dict_list[i]
            i = i + 1
            for key, value in encoded_dict.items():
                X_resampled_copy[:, col] = np.where(np.round(X_resampled_copy[:, col], 4) == np.round(
                    value * median_std_, 4), key, X_resampled_copy[:, col])

        for key, value in nan_dict.items():
            for item in value:
                X_resampled_copy[item, continuous_features_.size +
                                 key] = X_copy[item, continuous_features_.size + key]

        X_resampled = X_resampled_copy
        indices_reordered = np.argsort(
            np.hstack((continuous_features_, categorical_features_)))
        if sparse.issparse(X_resampled):
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]
        return X_resampled, y_resampled
'''
