from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from xgboost import XGBClassifier
from hyperopt import hp
import pandas as pd
import logging
from utils import (
    get_scaler,
    hyperopt_search,
    cross_validate_model,
)

LOGGER = logging.getLogger(__name__)


def run_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Run baseline model."""
    LOGGER.info("Running  baseline model..")
    dummy_clf = DummyClassifier(strategy="most_frequent")

    dummy_clf.fit(X_train, y_train)

    if config["test"]:
        y_pred = dummy_clf.predict(X_test)
        score = accuracy_score(y_pred=y_pred, y_true=y_test)

        LOGGER.info(f"baseline model has an accuracy of {score}")


def run_gaussian(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Any:
    LOGGER.info("Finding best gaussian model..")
    kernel = 1.0 * RBF(1.0)
    model = make_pipeline(get_scaler(config), GaussianProcessClassifier(kernel=kernel))

    mean_cross_val_score = cross_validate_model(model, X_train, y_train)
    LOGGER.info(f"Gaussian classifier cross validation score: {mean_cross_val_score}")

    if config["test"]:
        print(classification_report(model.predict(X_test), y_test))

    return model


def run_decisiontree(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Any:
    LOGGER.info("Finding best decision tree..")
    search_space = {
        "type": "decision_tree",
        "max_depth": hp.uniformint("max_depth", 2, 15),
        "min_samples_split": hp.uniformint("n_estimators", 2, 20),
        # "class_weight": hp.choice("class_weight", ["balanced"]),
    }

    best_params = hyperopt_search(X_train, y_train, search_space, config)
    model = make_pipeline(DecisionTreeClassifier(**best_params))

    mean_cross_val_score = cross_validate_model(model, X_train, y_train)
    LOGGER.info(f"Decision tree cross validation score: {mean_cross_val_score}")

    if config["test"]:
        print(classification_report(model.predict(X_test), y_test))

    return model


def run_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Any:
    """ Finds optimal model parameters for a KNN classifier, evaluates model and return model object."""
    LOGGER.info("Finding best knn..")
    search_space = {
        "type": "knn",
        "n_neighbors": hp.uniformint("n_neighbors", 2, 15),
        "weights": hp.choice("weights", ["uniform", "distance"]),
    }

    best_params = hyperopt_search(X_train, y_train, search_space, config)
    model = make_pipeline(get_scaler(config), KNeighborsClassifier(**best_params))

    mean_cross_val_score = cross_validate_model(model, X_train, y_train)
    LOGGER.info(f"KNN cross validation score: {mean_cross_val_score}")

    if config["test"]:
        print(classification_report(model.predict(X_test), y_test))

    return model


def run_svm_vote(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Any:
    LOGGER.info("Finding best svm..")
    search_space = {
        "type": "svm",
        "C": hp.lognormal("C", 0, 100.0),
        "gamma": hp.lognormal("gamma", 0, 1.0),
        "kernel": hp.choice("kernel", ["rbf"]),
    }

    best_params = hyperopt_search(X_train, y_train, search_space, config)
    model = make_pipeline(
        get_scaler(config),
        SVC(**best_params, class_weight="balanced", probability=True),
    )

    mean_cross_val_score = cross_validate_model(model, X_train, y_train)
    LOGGER.info(f"SVM cross validation score: {mean_cross_val_score}")

    if config["test"]:
        print(classification_report(model.predict(X_test), y_test))

    return model


def build_voting_classifier(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    LOGGER.info("building a voting classifier..")

    knn_clf = run_decisiontree(X_train, X_test, y_train, y_test, config)
    dt_clf = run_knn(X_train, X_test, y_train, y_test, config)
    svm_clf = run_svm_vote(X_train, X_test, y_train, y_test, config)
    gaussian_clf = run_gaussian(X_train, X_test, y_train, y_test, config)

    voting_clf = VotingClassifier(
        estimators=[
            ("gaussian", gaussian_clf),
            ("knn", knn_clf),
            ("dt", dt_clf),
            ("svm", svm_clf),
        ],
        voting="soft",
    ).fit(X_train, y_train)

    mean_cross_val_score = cross_validate_model(voting_clf, X_train, y_train)
    LOGGER.info(f"Voting classifier cross validation score: {mean_cross_val_score}")

    if config["test"]:
        print(classification_report(voting_clf.predict(X_test), y_test))


def run_svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Any, Any]:
    LOGGER.info("Running SVM..")
    if config["find_optimal_model"]:
        search_space = {
            "type": "svm",
            "C": hp.lognormal("C", 0, 10.0),
            "gamma": hp.lognormal("gamma", 0, 1.0),
            "kernel": hp.choice("kernel", ["rbf"]),
        }

        best_params = hyperopt_search(X_train, y_train, search_space, config)
        model = make_pipeline(SVC(**best_params)).fit(X_train, y_train)

        mean_cross_val_score = cross_validate_model(model, X_train, y_train)
        LOGGER.info(f"SVM classifier cross validation score: {mean_cross_val_score}")

    else:
        model = make_pipeline(SVC(**config["models"]["svm"])).fit(X_train, y_train)
        mean_cross_val_score = cross_validate_model(model, X_train, y_train)
        LOGGER.info(f"SVM cross validation score: {mean_cross_val_score}")
        if config["test"]:
            print(classification_report(model.predict(X_test), y_test))

    if config["test"]:
        y_pred = model.predict(X_test)
        score = accuracy_score(y_pred=y_pred, y_true=y_test)

        LOGGER.info(f"SVM has a test accuracy of {score}")

        return model, y_pred

    return model, None


def run_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Any, Any]:
    LOGGER.info("Running Random Forest model..")
    if config["find_optimal_model"]:
        search_space = {
            "type": "random_forest",
            "max_depth": hp.uniformint("max_depth", 2, 30),
            "n_estimators": hp.uniformint("n_estimators", 10, 1000),
            "max_features": hp.choice("max_features", ("auto", "sqrt", None)),
        }

        best_params = hyperopt_search(X_train, y_train, search_space, config)
        model = make_pipeline(RandomForestClassifier(**best_params))

        mean_cross_val_score = cross_validate_model(model, X_train, y_train)
        LOGGER.info(
            f"Random Forest classifier cross validation score: {mean_cross_val_score}"
        )

    else:
        model = make_pipeline(
            RandomForestClassifier(**config["models"]["random_forest"])
        ).fit(X_train, y_train)
        if config["test"]:
            print(classification_report(model.predict(X_test), y_test))

    if config["test"]:
        y_pred = model.predict(X_test)
        score = accuracy_score(y_pred=y_pred, y_true=y_test)

        LOGGER.info(f"Random forest model has a train accuracy of {score}")

        return model, y_pred

    return model, None


def run_gradient_boosting_classifier(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Any, Any]:
    LOGGER.info("Running Gradient boosting classifier..")
    if config["find_optimal_model"]:
        search_space = {
            "type": "gradient_boosting",
            "max_depth": hp.uniformint("max_depth", 2, 15),
            "n_estimators": hp.uniformint("n_estimators", 50, 300),
            "max_features": hp.choice("max_features", ("auto", "sqrt", None)),
            "learning_rate": hp.quniform("learning_rate", 0.025, 0.5, 0.025),
        }

        best_params = hyperopt_search(X_train, y_train, search_space, config)
        model = make_pipeline(GradientBoostingClassifier(**best_params)).fit(
            X_train, y_train
        )

        mean_cross_val_score = cross_validate_model(model, X_train, y_train)
        LOGGER.info(
            f"Gradient boosting classifier cross validation score: {mean_cross_val_score}"
        )

    else:
        model = make_pipeline(
            GradientBoostingClassifier(**config["models"]["gradient_boosting"])
        ).fit(X_train, y_train)
        if config["test"]:
            print(classification_report(model.predict(X_test), y_test))

    if config["test"]:
        y_pred = model.predict(X_test)
        score = accuracy_score(y_pred=y_pred, y_true=y_test)

        LOGGER.info(f"The gradient boosting classifier has a train accuracy of {score}")

        return model, y_pred

    return model, None


def run_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Any, Any]:
    LOGGER.info("Running XGboost..")
    if config["find_optimal_model"]:

        search_space = {
            "type": "xgboost",
            "max_depth": hp.choice("max_depth", range(5, 15, 1)),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.5, 0.01),
            "n_estimators": hp.choice("n_estimators", range(20, 205, 5)),
            "gamma": hp.quniform("gamma", 0, 0.50, 0.01),
            "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
            "subsample": hp.quniform("subsample", 0.1, 1, 0.01),
            # "colsample_bytree": hp.quniform("colsample_bytree", 0.1, 1.0, 0.01)
        }

        best_params = hyperopt_search(X_train, y_train, search_space, config)
        model = make_pipeline(XGBClassifier(**best_params)).fit(X_train, y_train)

        mean_cross_val_score = cross_validate_model(model, X_train, y_train)
        LOGGER.info(
            f"XGboost classifier cross validation score: {mean_cross_val_score}"
        )

    else:
        model = XGBClassifier(**config["models"]["xgboost"]).fit(X_train, y_train)
        if config["test"]:
            print(classification_report(model.predict(X_test), y_test))

    if config["test"]:
        y_pred = model.predict(X_test)
        score = accuracy_score(y_pred=y_pred, y_true=y_test)

        LOGGER.info(f"Xgboost has a train accuracy of {score}")

        return model, y_pred

    return model, None
