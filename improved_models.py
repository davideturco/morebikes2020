from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.metrics import mean_absolute_error
from phase3 import prepare_general_dataset
from sklearn.model_selection import train_test_split


class ImprovedLinearRegressor(BaseEstimator):

    def __init__(self, model_name):
        self.model_name = model_name
        self.intercept, self.coefficients, self.features = self.__prepare_model__(self.model_name)
        self._estimator_type = "regressor"

    def fit(self, x_train, y_train):
        return self

    def predict(self, x_test):
        x_test = x_test.filter(items=self.features)
        prediction = x_test.apply(lambda row: self.intercept + row.dot(self.coefficients), axis=1).astype('int64').values
        return prediction

    def __prepare_model__(self, model_path):
        pretrained_model = pd.read_csv(model_path + "_averaged.csv")
        features = pretrained_model['feature']

        # weights
        intercept = pretrained_model['weight'].values[0]
        weights = pretrained_model['weight'][1:]
        features_used = features.values[1:]
        # reindex to perform series multiplication
        weights.index = features_used

        return intercept, weights, features_used


if __name__ == '__main__':

    #df = prepare_general_dataset()
    df = pd.read_csv('df.csv')
    df_train, df_test = train_test_split(df)
    X_train = df_train.loc[:, df_train.columns != 'bikes']
    #X_train = X_train.filter(items=features_used)
    Y_train = df_train['bikes']
    X_test = df_test.loc[:, df_test.columns != 'bikes']
    #X_test = X_test.filter(items=features_used)
    Y_test = df_test['bikes']

    model = ImprovedLinearRegressor(model='short_full')
    model.fit(X_train,Y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(predictions, Y_test)
    print(mae)
