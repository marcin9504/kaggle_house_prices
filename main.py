from warnings import filterwarnings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200)

from sklearn.exceptions import DataConversionWarning

filterwarnings(action='ignore', category=DataConversionWarning)


def main():
    train_X, train_y, test_X, test_y = read_and_prepare_data(clean=False)

    # test_classifiers(train_X, train_y)
    clf = grid_search_over_estimator(MLPRegressor(early_stopping=True, max_iter=100000),
                                     {"hidden_layer_sizes": [(5, 5), (30, 30), (100, 100), (200, 200), (400, 400)]},
                                     train_X, train_y)

    classify(clf, test_X, test_y)


def classify(clf, test_X, test_y):
    out = clf.predict(test_X)
    print("Id,SalePrice")
    for idx, o in enumerate(out):
        print(test_y[idx], ",", o, sep="")


def grid_search_over_estimator(estimator, parameters, train_X, train_y):
    clf = GridSearchCV(estimator, parameters, cv=10, n_jobs=-1)
    clf.fit(train_X, train_y)

    print(clf.cv_results_['mean_test_score'])
    return clf.best_estimator_


def test_classifiers(train_X, train_y):
    classifiers = [
        # LogisticRegression(solver="lbfgs", max_iter=2000),
        LinearRegression(),
        # MLPRegressor(hidden_layer_sizes=(5, 5), max_iter=100000),
        # MLPRegressor(hidden_layer_sizes=(20, 5), max_iter=100000),
    ]
    for clf in classifiers:
        # clf.fit(train_X, train_y)
        # score = clf.score(train_X, train_y)
        score = np.mean(cross_val_score(clf, train_X, train_y, cv=5, n_jobs=-1))
        print(score)


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def read_and_prepare_data(clean=False):
    df_train = pd.read_csv('train.csv', sep=",")

    df_test = pd.read_csv('test.csv', sep=",")
    # print(df_train.head())
    # print(df_train.describe())

    column_names = ["Id", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YrSold", "SalePrice"]
    df_train = df_train[column_names[1:]]
    df_test = df_test[column_names[:-1]]
    # return

    column_names = column_names[1:-1]

    # for col in column_names:
    #     df_train = hot_encode_column(col, df_train)
    #     df_test = hot_encode_column(col, df_test)

    # column_names = ["Pclass", "Age", "Fare", "Family_size"]
    for col in column_names:
        scaler = StandardScaler()
        scaler.fit(df_train[[col]])
        scale(col, df_train, scaler)
        scale(col, df_test, scaler)

    # column_names = ["Name", "Ticket", "Cabin", "Embarked", "Sex_female", "SibSp", "Parch"]
    # df_train = df_train.drop(labels=column_names, axis=1)
    # df_test = df_test.drop(labels=column_names, axis=1)
    # print(df_train.head())
    # print(df_test.head())

    if clean:
        clean_dataset(df_train)
        clean_dataset(df_test)
    else:
        df_train = df_train.fillna(0)
        df_test = df_test.fillna(0)

    train_y_labels = df_train[["SalePrice"]]  # saleprice
    train_x_labels = df_train.columns[:-1]  # all but saleprice

    test_y_labels = df_test[["Id"]]  # id only
    test_x_labels = df_test.columns[1:]  # all but id

    X = df_train.filter(train_x_labels)
    y = df_train.filter(train_y_labels)
    train_X = X.values
    train_y = y.values.ravel()

    X = df_test.filter(test_x_labels)
    y = df_test.filter(test_y_labels)
    test_X = X.values
    test_y = y.values.ravel()

    return train_X, train_y, test_X, test_y


def scale(col, df, scaler):
    columns = df[[col]]
    scaled_values = scaler.fit_transform(columns)
    df[col] = scaled_values


def hot_encode_column(column_name, df):
    new_cols = pd.get_dummies(df[column_name], prefix=column_name)
    df = df.drop(column_name, axis=1)
    df = df.join(new_cols)
    return df


if __name__ == "__main__":
    main()
