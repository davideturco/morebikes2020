import pandas as pd
import glob
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from datetime import datetime


################################
## Utils
################################

def load_individual(path):
    """
    Loads all the csv files in a given directory and store them in a list of separate dataframes
    """
    df = []
    for file in glob.glob(path + "/*.csv"):
        dataframe = pd.read_csv(file)
        df.append(dataframe)
    return df


def load_general(path):
    """
    Loads all the csv files in a given directory and concatenate them into a single datafram
    """
    df = load_individual(path)
    general_df = pd.concat(df, ignore_index=True)
    return general_df


def has_nan(dataframe):
    """
    Return true if dataframe has missing values (e.g. NaN) and counts how many missing value each feature has
    """
    is_nan = dataframe.isnull().values.any()
    no_nan = dataframe.isnull().sum()
    # is_infinite = np.all(np.isfinite(dataframe))
    return is_nan, no_nan  # , is_infinite


def get_time(timestamp):
    """
    Returns UTC (classic) time from UNIX timestamp
    """
    time = datetime.utcfromtimestamp(timestamp)
    year, month, day, hour, minute = time.year, time.month, time.day, time.hour, time.minute

    return year, month, day, hour, minute


def drop_nan_bikes(dataset):
    """
    Drops rows where there is a null value for bikes (e.g. NaN)
    """
    new_dataset = dataset.drop(dataset[dataset['bikes'].isnull()].index)
    return new_dataset


def day_transform(dataset):
    """
    Function that replaces strings of weekdays with a numerical equivalent.
    """

    dataset = dataset.replace(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        [1, 2, 3, 4, 5, 6, 7])

    return dataset


def nan_impute(dataset):
    """
    Function that replaces all the NaN appearing in the dataset with the median value along each column.
    """
    #replacer = SimpleImputer(missing_values=np.nan, strategy='median')
    replacer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='distance')
    new_dataset = pd.DataFrame(replacer.fit_transform(dataset), columns=dataset.columns)

    return new_dataset


def var_transform(dataset, threshold=0.0):
    """
    Function that removes features with variance below a certain threshold (default: threshold=0)
    """
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
