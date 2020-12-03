import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


def day_transform(dataset):
    """
    Function that replaces strings of weekdays with a numerical equivalent.
    """

    transformed = dataset.replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                  [1, 2, 3, 4, 5, 6, 7], inplace=True)

    return transformed


def nan_imputer(dataset):
    """
    Function that replaces all the NaN appearing in the dataset with the median value along each column.
    """
    replacer = SimpleImputer(missing_values=np.nan, strategy='median')
    new_dataset = pd.DataFrame(replacer.fit_transform(dataset), columns=dataset.columns)

    return new_dataset


def var_transform(dataset, threshold=0.0):
    """
    Function that removes features with variance below a certain threshold (default: threshold=0)
    """
    # if threshold is None:
    #     threshold = 0.0
    selector = VarianceThreshold(threshold)
    selector.fit_transform(dataset)
    new_dataset = dataset[dataset.columns[selector.get_support(indices=True)]]

    return new_dataset


def correl(dataset):
    """
    Function that computes the correlation matrix for a given dataset.

    :param dataset: Pandas dataframe
    :return: Correlation matrix (Pandas Dataframe)
    """
    corr_matrix = dataset.corr().abs()
    return corr_matrix


def high_correl(dataset, threshold):
    """
    Function that removes features with correlation higher that the given threshold. By default, the second column in
    order is dropped.
    :param dataset: Pandas dataframe
    :param threshold: Threshold, decimal number between 0 and 1
    :returns list of columns which have been dropped
    """

    corr_matrix = correl(dataset)
    to_drop = []

    # only consider upper triangle of correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                name = corr_matrix.columns[i]
                to_drop.append(name)

                if name in dataset.columns:
                    del dataset[name]

    return to_drop
