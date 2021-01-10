from data_upload import *
from sklearn.metrics import mean_absolute_error
import glob
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

MODELS = {'short': 'Models/Models/model_station_*_rlm_short.csv',
          'short_temp': 'Models/Models/model_station_*_rlm_short_temp.csv',
          'full': 'Models/Models/model_station_*_rlm_full.csv',
          'full_temp': 'Models/Models/model_station_*_rlm_full_temp.csv',
          'short_full': 'Models/Models/model_station_*_rlm_short_full.csv',
          'short_full_temp': 'Models/Models/model_station_*_rlm_short_full_temp.csv'

          }


def plot_scores_1(scores):
    x = np.arange(0, len(scores[0]))
    # plt.xticks(np.arange(201, 276))
    plt.xlabel('Station')
    plt.ylabel('MAE')
    plt.plot(x, scores[0], '--', label='short')
    plt.plot(x, scores[1], label='short_temp')
    plt.plot(x, scores[2], label='full')
    plt.plot(x, scores[3], label='full_temp')
    plt.plot(x, scores[4], label='short_full')
    plt.plot(x, scores[5], label='short_full_temp')
    plt.legend()
    plt.show()


def plot_scores_2(scores):
    x = np.arange(0, len(scores[1][1]))
    plt.xlabel('Station')
    plt.ylabel('MAE')
    plt.plot(x, scores[0][1], '--', label=scores[0][0])
    plt.plot(x, scores[1][1], label=scores[1][0])
    plt.plot(x, scores[2][1], label=scores[2][0])
    plt.plot(x, scores[3][1], label=scores[3][0])
    plt.plot(x, scores[4][1], label=scores[4][0])
    plt.plot(x, scores[5][1], label=scores[5][0])
    plt.xticks(np.arange(0, 75, 3), np.arange(201, 276, 3), rotation=45)
    plt.legend()
    plt.show()


def get_averaged_models():
    """For each pre-trained model type, this function calculates the mean intercept and feature weights over the
    stations the model was trained on and stores them in .csv files """
    global features
    for key, value in MODELS.items():
        intercepts = []
        weights_list = []
        for file in tqdm(glob.glob(value)):
            model = pd.read_csv(file)
            features = model['feature']

            # weights
            intercept = model['weight'].values[0]
            weights = model['weight'][1:]
            intercepts.append(intercept)
            weights_list.append(weights)

        new_intercept = mean(intercepts)
        averaged_weights = [sum(col) / len(col) for col in zip(*weights_list)]
        df1 = pd.DataFrame([['Intercept', new_intercept]], columns=["feature", "weight"])
        df2 = pd.DataFrame({"feature": features[1:].values, "weight": averaged_weights})
        new_model = pd.concat([df1, df2], ignore_index=True)
        new_model.to_csv(key + '_averaged.csv', index=False)


def evaluate_models_2():
    """Evaluate the 6 averaged models on each of the 75 stations in the training set and plot the mean absolute
    errors """
    df = prepare_individual_datasets()
    scores = []
    print("Starting evaluation...")
    for model in glob.glob('*_averaged.csv'):
        averaged_model = pd.read_csv(model)
        features = averaged_model['feature']

        # weights
        intercept = averaged_model['weight'].values[0]
        weights = averaged_model['weight'][1:]
        features_used = features.values[1:]
        # reindex to perform series multiplication
        weights.index = features_used

        temp_scores = []
        for station in df:
            X = station.loc[:, station.columns != 'bikes']
            Y = station['bikes']
            X = X.filter(items=features_used)
            predictions = X.apply(lambda row: intercept + row.dot(weights), axis=1).astype('int64')
            temp_scores.append(mean_absolute_error(predictions, Y))
        name = model.split('_averaged')[0]
        scores.append((name, temp_scores))
        print(f'Accuracy of model {name} is {mean(temp_scores)}\n')
    plot_scores_2(scores)


def evaluate_models_1():
    """Evaluate all the 1200 models on the general dataframe and plots the average score per each model type  """
    df = prepare_general_dataset()
    X = df.loc[:, df.columns != 'bikes']
    Y = df['bikes']
    scores = []

    for key, value in MODELS.items():
        X = df.loc[:, df.columns != 'bikes']
        temp_scores = []
        for file in tqdm(glob.glob(value)):
            model = pd.read_csv(file)
            features = model['feature']

            # weights
            intercept = model['weight'].values[0]
            weights = model['weight'][1:]

            features_used = features.values[1:]
            X = X.filter(items=features_used)

            # reindex to perform series multiplication
            weights.index = X.iloc[1, :].index

            predictions = X.apply(lambda row: intercept + row.dot(weights), axis=1).astype('int64')
            temp_scores.append(mean_absolute_error(predictions, Y))

        print(f'\nModel {key} performance:')
        print(mean(temp_scores))
        print(min(temp_scores))
        print('\n')

        scores.append(temp_scores)

    with open('scores.txt', 'wb') as file:
        pickle.dump(scores, file)

    plot_scores_1(scores)


if __name__ == '__main__':
    # get_averaged_models()
    evaluate_models_1()
