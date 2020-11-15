from utils import (
    split_data,
    read_challenge_data,
    process_and_filter_data,
    process_evaluate_data,
)
from models import (
    run_baseline,
    run_random_forest,
    run_gradient_boosting_classifier,
    run_xgboost,
)
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
)
from pathlib import Path
import pandas as pd
import logging
import yaml
import sys

LOGGER = logging.getLogger(__name__)


def main():
    df_train, df_evaluate = read_challenge_data()
    df_train = process_and_filter_data(df_train, config)
    df_evaluate = process_evaluate_data(df_evaluate, config)

    X_train, X_test, y_train, y_test = split_data(df_train, config)

    run_baseline(X_train, X_test, y_train, y_test, config)

    gb_clf, y_pred_gb = run_gradient_boosting_classifier(
        X_train, X_test, y_train, y_test, config
    )
    rf_clf, y_pred_rf = run_random_forest(X_train, X_test, y_train, y_test, config)
    xgb_clf, y_pred_xgb = run_xgboost(X_train, X_test, y_train, y_test, config)

    # Voting classifier.
    voting_clf = VotingClassifier(
        estimators=[("rf", rf_clf), ("gb", gb_clf), ("xgb", xgb_clf)], voting="soft"
    ).fit(X_train, y_train)
    stratified_shuffle_split = StratifiedKFold(n_splits=10)
    cross_val_score_ = cross_val_score(
        voting_clf, X_train, y_train, cv=stratified_shuffle_split
    ).mean()
    LOGGER.info(f"Voting classifier cross validation score: {cross_val_score_}")
    if config["test"]:
        print(classification_report(voting_clf.predict(X_test), y_test))

    final_model = voting_clf
    final_predictions = final_model.predict(df_evaluate)

    pd.DataFrame(final_predictions).to_csv("data/101617.txt", index=False, header=False)


if __name__ == "__main__":
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
