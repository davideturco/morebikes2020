import os
from joblib import load
import pandas as pd
from pre_processing import *
from feature_selection import *

PATH_TO_DATA = './test.csv'
PATH_TO_SUBMISSION = './predictions.csv'


def load_model():
    model = load('stacking.joblib')
    return model


def preprocess_data():
    dataframe = pd.read_csv(PATH_TO_DATA)
    # dataframe = day_transform(dataframe)
    dataframe = pd.get_dummies(dataframe, columns=['weekday'])
    dataframe = var_transform(dataframe)
    #to_drop = high_correl(dataframe, 0.95)
    id_dataframe = dataframe['Id']

    dataframe['isWorkingTime'] = dataframe.apply(feat, axis=1)

    dataframe['occupancy_3h_ago'] = dataframe.apply(occupancy, axis=1)
    dataframe = dataframe.drop(['Id', 'year', 'month', 'precipitation.l.m2', 'day', 'windMeanSpeed.m.s',
                                'short_profile_3h_diff_bikes', 'short_profile_bikes'], axis=1)

    scaler = load('scaler.joblib')
    dataframe = pd.DataFrame(scaler.transform(dataframe), columns=dataframe.columns)
    return id_dataframe, dataframe


def main():
    model = load_model()
    id_dataframe, data = preprocess_data()

    predictions = pd.DataFrame(model.predict(data), columns=['bikes'], dtype="int64")
    submission = pd.concat([id_dataframe, predictions], axis=1)
    submission.to_csv(PATH_TO_SUBMISSION, index=False)


if __name__ == "__main__":
    main()
