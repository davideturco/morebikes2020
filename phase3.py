from data_upload import *
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MODELS = {'short': 'short_averaged.csv',
          'short_temp': 'short_temp_averaged.csv',
          'full': 'full_averaged.csv',
          'full_temp': 'full_temp_averaged.csv',
          'short_full': 'short_full_averaged.csv',
          'short_full_temp': 'short_full_temp_averaged.csv'

          }


def prepare_model(model_path):
    model = pd.read_csv(model_path)
    features = model['feature']

    # weights
    intercept = model['weight'].values[0]
    weights = model['weight'][1:]
    features_used = features.values[1:]
    # reindex to perform series multiplication
    weights.index = features_used

    return intercept, weights, features_used


def fit_ridge(train_x, train_y):
    model = Ridge(alpha=0.01)
    model.fit(train_x, train_y)
    return model.intercept_, model.coef_


def average_coefficients(intercept_ridge, coefficients_ridge, intercept_pretrained, coefficients_pretrained):
    averaged_intercept = (intercept_ridge + 12 * intercept_pretrained) / 13
    averaged_coefficients = (coefficients_ridge.add(12 * coefficients_pretrained)) / 13
    return averaged_intercept, averaged_coefficients


def integrated_model(X, intercept, coefficients):
    predictions = X.apply(lambda row: intercept + row.dot(coefficients), axis=1).astype('int64')
    return predictions


def main():
    df = prepare_general_dataset()
    df_train, df_test = train_test_split(df)

    for key, value in MODELS.items():
        intercept_ridge, weights, features_used = prepare_model(value)
        X_train = df_train.loc[:, df_train.columns != 'bikes']
        X_train = X_train.filter(items=features_used)
        Y_train = df_train['bikes']
        X_test = df_test.loc[:, df_test.columns != 'bikes']
        X_test = X_test.filter(items=features_used)
        Y_test = df_test['bikes']

        intercept, coefficients = fit_ridge(X_train, Y_train)
        coefficients = pd.Series(coefficients, index=features_used)
        q, a = average_coefficients(intercept_ridge, weights, intercept, coefficients)
        predictions = integrated_model(X_test, q, a)
        mae = mean_absolute_error(predictions, Y_test)
        print(f"\nThe MAE for model {key} is {mae}\n")


if __name__ == '__main__':
    main()
