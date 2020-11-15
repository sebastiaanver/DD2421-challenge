from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from hyperopt import fmin, rand, tpe, space_eval, STATUS_OK
from sklearn.svm import SVC
import pandas as pd
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple, List
import logging
import time


LOGGER = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    data = load_breast_cancer()
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]
    return df


def get_scaler(config: Dict[str, Any]) -> None:
    if config["scaler"] == "standard":
        return StandardScaler()
    elif config["scaler"] == "minmax":
        return MinMaxScaler()
    elif config["scaler"] == "robust":
        return RobustScaler()
    else:
        LOGGER.info("Name of scaler is not known, StandardScaler is selected.")
        return StandardScaler()


def oversample_data(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oversample = RandomOverSampler(sampling_strategy="minority")
    X, y = oversample.fit_resample(X_train, y_train)

    return X, y


def split_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Any, pd.DataFrame, Any]:
    X = df.loc[:, df.columns != config["target_name"]]
    y = df[config["target_name"]]

    if config["test"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        return X_train, X_test, y_train, y_test

    else:
        return X, None, y, None


def perform_grid_serach(
    classifier: Any, param_grid, X_train: pd.DataFrame, y_train: pd.DataFrame
) -> Any:
    """Perform a gridsearch and return the best estimator."""
    pipe = make_pipeline(SimpleImputer(), classifier)
    stratified_shuffle_split = StratifiedKFold(n_splits=10)
    grid = GridSearchCV(pipe, param_grid, cv=stratified_shuffle_split).fit(
        X_train, y_train
    )
    LOGGER.info(f"best model parameters: {grid.best_params_}")
    LOGGER.info(f"best cross validation score: {grid.best_score_}")

    return grid.best_estimator_


def hyperopt_search(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameter_space,
    config: Dict[str, Any],
) -> Dict:
    """Hyperparameter tuning for the given feature set.

    Returns:
        Set of optimal parameters.
    """

    def objective(params):
        classifier_type = params["type"]
        del params["type"]
        if classifier_type == "gradient_boosting":
            clf = make_pipeline(
                get_scaler(config), GradientBoostingClassifier(**params)
            )
        elif classifier_type == "xgboost":
            clf = XGBClassifier(**params)
        elif classifier_type == "gaussian":
            clf = make_pipeline(get_scaler(config), GaussianProcessClassifier(**params))
        elif classifier_type == "decision_tree":
            clf = DecisionTreeClassifier(**params, n_jobs=-1)
        elif classifier_type == "random_forest":
            clf = RandomForestClassifier(**params, n_jobs=-1)
        elif classifier_type == "svm":
            clf = make_pipeline(get_scaler(config), SVC(**params))
        elif classifier_type == "knn":
            clf = make_pipeline(
                get_scaler(config), KNeighborsClassifier(**params, n_jobs=-1)
            )
        else:
            return 0

        stratified_shuffle_split = StratifiedKFold(n_splits=10)
        accuracy = cross_val_score(
            clf, X_train, y_train, cv=stratified_shuffle_split, n_jobs=-1
        ).mean()

        return {"loss": -accuracy, "status": STATUS_OK}

    LOGGER.info("Start hyperopt search.")
    start = time.time()
    best_params = fmin(
        fn=objective,
        space=parameter_space,
        algo=rand.suggest,  # Can be change to rand.suggest or tpe.suggest or atpe.suggest.
        max_evals=config["evals"],
        timeout=config["timeout"],
    )

    LOGGER.info("Hyperopt search took: %.2f s", time.time() - start)

    best_params = space_eval(parameter_space, best_params)
    LOGGER.info(best_params)
    del best_params["type"]
    return best_params


def read_challenge_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv("data/TrainOnMe.csv")
    df_evaluate = pd.read_csv("data/EvaluateOnMe.csv")

    return df_train, df_evaluate


def process_evaluate_data(
    df_evaluate: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    df_evaluate["x5"].replace({True: 1.0, False: 0.0}, inplace=True)
    if config["labels"] != "one_hot":
        df_evaluate["x6"].replace(
            {"A": 5.0, "B": 4.5, "C": 4.0, "D": 3.5, "E": 3.0, "F": 0.0, "Fx": 1.5},
            inplace=True,
        )
    else:
        df_evaluate = pd.concat(
            [df_evaluate, pd.get_dummies(df_evaluate["x6"])], axis=1
        )
        df_evaluate.drop(["x6"], axis=1, inplace=True)
    df_evaluate.drop(config["dropped_columns"], axis=1, inplace=True)

    return df_evaluate


def process_and_filter_data(
    df_train: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:

    # Filter out outliers and weird data.
    df_train = df_train[df_train["x8"] < 100000]
    df_train = df_train[df_train["x7"] > -800]
    df_train = df_train[df_train["x5"] != "?"]
    df_train.dropna(inplace=True)

    # Change categorical to numerical values.
    df_train["x5"].replace({"True": 1.0, "False": 0.0}, inplace=True)
    if config["labels"] != "one_hot":
        df_train["x6"].replace(
            {"A": 5.0, "B": 4.5, "C": 4.0, "D": 3.5, "E": 3.0, "F": 0.0, "Fx": 1.5},
            inplace=True,
        )
    else:
        df_train = pd.concat([df_train, pd.get_dummies(df_train["x6"])], axis=1)
        df_train.drop(["x6"], axis=1, inplace=True)

    # Transform to numeric columns.
    df_train["x1"] = pd.to_numeric(df_train["x1"])
    df_train["x2"] = pd.to_numeric(df_train["x2"])

    df_train.drop(config["dropped_columns_train"], axis=1, inplace=True)

    return df_train


def rfecv_selector(model, X_train: pd.DataFrame, y_train: pd.DataFrame,) -> List[str]:
    """Recursive feature elimination.

    Returns:
        Set of selected features.
    """
    LOGGER.info("Start RFECV feature selection.")
    start = time.time()

    stratified_shuffle_split = StratifiedKFold(n_splits=10)
    selector = RFECV(
        model, step=1, min_features_to_select=5, cv=stratified_shuffle_split, n_jobs=-1,
    )
    selector = selector.fit(X_train, y_train)
    LOGGER.info("RFECV feature selection took: %.2f s", time.time() - start)

    LOGGER.info(
        "Set of features selected by RFECV: %s", X_train.columns[selector.support_],
    )
    return X_train.columns[selector.support_]


def cross_validate_model(
    model: Any, X_train: pd.DataFrame, y_train: pd.DataFrame
) -> float:
    stratified_shuffle_split = StratifiedKFold(n_splits=10)
    avg_cross_val_score = cross_val_score(
        model, X_train, y_train, cv=stratified_shuffle_split
    ).mean()

    return avg_cross_val_score
