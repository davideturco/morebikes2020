import pandas as pd
from pre_processing import *
from feature_selection import *
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def prepare_general_dataset():
    df = load_general('Train/Train')
    df = drop_nan_bikes(df)
    df = pd.get_dummies(df, columns=['weekday'])
    df = var_transform(df)
    df = nan_impute(df)

    df.to_csv('df.csv', index=False)

    return df


def prepare_model():
    model = pd.read_csv('full_temp_averaged.csv')
    features = model['feature']

    # weights
    intercept = model['weight'].values[0]
    weights = model['weight'][1:]
    features_used = features.values[1:]
    # reindex to perform series multiplication
    weights.index = features_used

    return intercept, weights, features_used


def fit_ridge(X_train, Y_train):
    model = Ridge()
    model.fit(X_train, Y_train)
    return model.intercept_, model.coef_


def average_coefficients(intercept_ridge, coefficients_ridge, intercept_pretrained, coefficients_pretrained):
    # TODO: check that this actually returns averaged coefficients!!
    averaged_intercept = (intercept_ridge + 12 * intercept_pretrained) / 13
    averaged_coefficients = (coefficients_ridge.add(12 * coefficients_ridge)) / 13
    return averaged_intercept, averaged_coefficients


def integrated_model(X, intercept, coefficients):
    predictions = X.apply(lambda row: intercept + row.dot(coefficients), axis=1).astype('int64')
    return predictions


def main():
    df = prepare_general_dataset()
    intercept_ridge, weights, features_used = prepare_model()
    df_train, df_test = train_test_split(df)
    X_train = df_train.loc[:, df_train.columns != 'bikes']
    X_train = X_train.filter(items=features_used)
    Y_train = df_train['bikes']
    intercept, coefficients = fit_ridge(X_train, Y_train)
    coefficients = pd.Series(coefficients, index=features_used)
    q, a = average_coefficients(intercept_ridge, weights, intercept, coefficients)
    predictions = integrated_model(X_train, q, a)
    mae = mean_absolute_error(predictions, Y_train)
    print(mae)
    # print(intercept)
    # print(coefficients)


if __name__ == '__main__':
    main()
