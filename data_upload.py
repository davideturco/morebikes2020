from pre_processing import *
from feature_selection import *
import os
import pickle


def prepare_general_dataset():
    """Uploads the general dataset and saves them in a csv file"""
    if os.path.isfile('df.csv'):
        df = pd.read_csv('df.csv')
        return df
    else:
        df = load_general('Train/Train')
        df = drop_nan_bikes(df)
        df = pd.get_dummies(df, columns=['weekday'])
        df = var_transform(df)
        df = nan_impute(df)
        df.to_csv('df.csv', index=False)
        return df


def prepare_individual_datasets():
    """Uploads the individual datasets and saves them in a list of dataframes, in a txt file"""
    if os.path.isfile('df_list.txt'):
        with open('df_list.txt', 'rb') as file:
            df = pickle.load(file)
        return df
    else:
        df = load_individual('Train/Train')
        df = list(map(drop_nan_bikes, df))

        # replaces days with numbers
        df = list(map(day_transform, df))

        # impute missing values
        df = list(map(nan_impute, df))

        # removes features with zero variance
        df = list(map(var_transform, df))

        with open('df_list.txt', 'wb') as file:
            pickle.dump(df, file)

        return df
