import pandas as pd
from pre_processing import *
from sklearn.metrics import mean_absolute_error
import glob
from statistics import mean
from tqdm import tqdm


# def linear_regression(row):
#     return intercept + row.mul(weights)


def main():
    df = pd.read_csv('df.csv')
    X = df.loc[:, df.columns != 'bikes']
    Y = df['bikes']
    scores = []
    for file in tqdm(glob.glob('Models/Models/model_station_*_rlm_full.csv')):
        model = pd.read_csv(file)
        features = model['feature']

        # weights
        intercept = model['weight'].values[0]
        weights = model['weight'][1:]

        features_used = features.values[1:]
        X = X.filter(items=features_used)

        # reindex to perform series multiplication
        weights.index = X.iloc[1,:].index

        predictions = X.apply(lambda row: intercept+row.dot(weights), axis=1).astype('int64')
        scores.append(mean_absolute_error(predictions, Y))

        X = df.loc[:, df.columns != 'bikes']

    print(mean(scores))
    print(min(scores))


if __name__ == '__main__':
    main()
