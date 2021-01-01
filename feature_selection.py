from sklearn.feature_selection import VarianceThreshold


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


def feat(row):
    """Adds a new binary feature that it is 1 if it a working time, 0 otherwise. Spanish working hours
    were used, i.e. working day is from 9 to 13 and from 16 to 20"""

    if row['weekday_Saturday'] != 1 and row['weekday_Sunday'] != 1:
        if 9. <= row['hour'] <= 13. or 16. <= row['hour'] <= 20.:
            return 1
    return 0


def occupancy(row):
    """Adds a new features that returns the percentage of docks occupied 3 hours ago (on a 0-1 scale"""
    return row['bikes_3h_ago'] / row['numDocks']
