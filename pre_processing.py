import pandas as pd
import glob
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
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


def day_dummies(dataset):
    new_dataset = pd.get_dummies(dataset, columns=['weekday'])
    return new_dataset


def nan_impute(dataset):
    """
    Function that replaces all the NaN appearing in the dataset with the median value along each column.
    """
    # replacer = SimpleImputer(missing_values=np.nan, strategy='median')
    #replacer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='distance')
    replacer = IterativeImputer()
    new_dataset = pd.DataFrame(replacer.fit_transform(dataset), columns=dataset.columns)

    return new_dataset
