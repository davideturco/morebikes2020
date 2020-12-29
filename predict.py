import os
from joblib import load
import pandas as pd
from pre_processing import *

PATH_TO_DATA = './test.csv'
PATH_TO_SUBMISSION = './predictions.csv'


def load_model():
    model = load('lgb.joblib')
    return model


def preprocess_data():
    dataframe = pd.read_csv(PATH_TO_DATA)
    # dataframe = day_transform(dataframe)
    dataframe = pd.get_dummies(dataframe, columns=['weekday'])
    dataframe = var_transform(dataframe)
    # to_drop = high_correl(dataframe, 0.95)
    id_dataframe = dataframe['Id']
    dataframe = dataframe.drop(['Id', 'year', 'month', 'relHumidity.HR', 'day', 'windMeanSpeed.m.s',
                                'short_profile_3h_diff_bikes', 'short_profile_bikes'], axis=1)

    scaler = load('scaler.joblib')
    dataframe = pd.DataFrame(scaler.transform(dataframe), columns=dataframe.columns)
    return id_dataframe, dataframe


def main():
    if not os.path.exists(PATH_TO_SUBMISSION):
        os.mknod(PATH_TO_SUBMISSION)
    model = load_model()
    id_dataframe, data = preprocess_data()

    predictions = pd.DataFrame(model.predict(data), columns=['bikes'], dtype="int32")
    submission = pd.concat([id_dataframe, predictions], axis=1)
    submission.to_csv(PATH_TO_SUBMISSION, index=False)


if __name__ == "__main__":
    main()
