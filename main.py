# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
from warnings import filterwarnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

pd.set_option("display.max_columns", None)
# pd.set_option("display.height", 1000)
# pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 200)

from sklearn.exceptions import DataConversionWarning

filterwarnings(action="ignore", category=DataConversionWarning)



def main():
    train = pd.read_csv("train.csv", sep=",")

    test = pd.read_csv("test.csv", sep=",")
    # print(train.head())
    # print(df_train.describe())

    train_ids = train["Id"]
    test_ids = test["Id"]

    train = train.drop("Id", axis=1)
    test = test.drop("Id", axis=1)
    # print(train.shape)
    # print(test.shape)

    fig, ax = plt.subplots()
    x = train["GrLivArea"]
    y = train["SalePrice"]
    ax.scatter(x, y)
    # plt.show()

    mask = train[
        (train["GrLivArea"] > 4500)
        & (train["SalePrice"] < 400000)
        ]
    train = train.drop(mask.index)

    fig, ax = plt.subplots()
    x = train["GrLivArea"]
    y = train["SalePrice"]
    ax.scatter(x, y)
    # plt.show()

    sns.distplot(train["SalePrice"], fit=norm)
    (mu, sigma) = norm.fit(train["SalePrice"])
    # print(mu, sigma)
    plt.ylabel("Frequency")
    plt.title("SalePrice distribution")
    fig = plt.figure()
    res = stats.probplot(train["SalePrice"], plot=plt)
    # plt.show()

    train["SalePrice"] = np.log1p(train["SalePrice"])

    sns.distplot(train["SalePrice"], fit=norm)
    (mu, sigma) = norm.fit(train["SalePrice"])
    # print(mu, sigma)
    plt.ylabel("Frequency")
    plt.title("SalePrice distribution")
    fig = plt.figure()
    res = stats.probplot(train["SalePrice"], plot=plt)
    # plt.show()

    num_train = train.shape[0]
    num_test = test.shape[0]
    y_train = train.SalePrice.values

    all_data = pd.concat((train, test), sort=True).reset_index(drop=True)
    # print(all_data.head())
    all_data = all_data.drop(["SalePrice"], axis=1)

    all_data_missing = all_data.isnull().sum() / len(all_data)
    all_data_missing = all_data_missing.sort_values(ascending=False)
    # print(all_data_missing)

    correlation_matrix = train.corr()
    plt.subplots()
    sns.heatmap(correlation_matrix)
    # plt.show()

    columns_to_fill_with_None = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish",
                                 "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                                 "BsmtFinType2", "MasVnrArea", "MSSubClass"]
    for column in columns_to_fill_with_None:
        all_data[column] = all_data[column].fillna("None")

    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"] \
        .transform(lambda x: x.fillna(x.median()))

    columns_to_fill_with_zero = ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                                 "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrType"]
    for column in columns_to_fill_with_zero:
        all_data[column] = all_data[column].fillna(0)

    columns_to_fill_with_most_frequent = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd",
                                          "SaleType"]
    for column in columns_to_fill_with_most_frequent:
        all_data[column] = all_data[column].fillna(all_data[column].mode()[0])

    all_data = all_data.drop(["Utilities"], axis=1)

    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    all_data_missing = all_data.isnull().sum() / len(all_data)
    all_data_missing = all_data_missing.sort_values(ascending=False)
    # print(all_data_missing)

    columns_to_change_to_categorial = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]
    for column in columns_to_change_to_categorial:
        all_data[column] = all_data[column].apply(str)

    columns_to_label_encode = ["FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond",
                               "ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual", "BsmtFinType1",
                               "BsmtFinType2", "Functional", "Fence", "BsmtExposure", "GarageFinish", "LandSlope",
                               "LotShape", "PavedDrive", "Street", "Alley", "CentralAir", "MSSubClass", "OverallCond",
                               "YrSold", "MoSold"]
    for column in columns_to_label_encode:
        label_encoder = LabelEncoder()
        label_encoder.fit((list(all_data[column].values)))
        all_data[column] = label_encoder.transform((list(all_data[column].values)))

    all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]

    numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_features = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # print(skewed_features)

    skewness = pd.DataFrame({"Skew": skewed_features})
    skewness = skewness[abs(skewness) > 0.75]

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    all_data = pd.get_dummies(all_data)

    train = all_data[:num_train]
    test = all_data[num_train:]

    models = {
        "lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1)),
        "ENet": make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)),
        "KRR": KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5),
        "GBoost": GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                            max_depth=4, max_features="sqrt",
                                            min_samples_leaf=15, min_samples_split=10,
                                            loss="huber", random_state=5),
        "model_xgb": xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                      learning_rate=0.05, max_depth=3,
                                      min_child_weight=1, n_estimators=2200,
                                      reg_alpha=0.4640, reg_lambda=0.8571,
                                      subsample=0.5213, silent=True,
                                      random_state=7, nthread=-1),
        "model_lgb": lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                       learning_rate=0.05, n_estimators=720,
                                       max_bin=55, bagging_fraction=0.8,
                                       bagging_freq=5, feature_fraction=0.2319,
                                       feature_fraction_seed=9, bagging_seed=9,
                                       min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
    }

    # for model_name in models:
    #     print(model_name)
    #     model = models[model_name]
    #     score = rmsle_cv(model, train, y_train)
    #     print(score.mean(), score.std())

    ensemble = AveragingModels(models=list(models.values()))
    # score = rmsle_cv(ensemble, train, y_train)
    # print(score.mean(), score.std())

    # print(np.shape(train.values), np.shape(test.values))


    ensemble.fit(train.values, y_train)
    predictions = ensemble.predict(test.values)
    predictions = np.expm1(predictions)

    final = pd.DataFrame()
    # print(test_ids, predictions)
    final["Id"] = test_ids
    final["SalePrice"] = predictions
    final.to_csv("submission.csv", index=False)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def rmsle_cv(model, train, y_train):
    n_folds = 2
    kf = KFold(n_folds, shuffle=True).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))
    return (rmse)


if __name__ == "__main__":
    main()
