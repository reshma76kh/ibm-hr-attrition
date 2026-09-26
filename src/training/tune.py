from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier


def get_tuning_models():

    models = {

        "SVC": (
            SVC(),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "sigmoid"],
                "gamma": ["scale", "auto"]
            }
        ),

        "KN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        ),

        "DT": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"]
            }
        ),

        "LR": (
            LogisticRegression(
                solver="liblinear",
                random_state=42
            ),
            {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"]
            }
        ),

        "RF": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        ),

        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 150, 200],
                "learning_rate": [0.01, 0.1, 0.5, 1.0]
            }
        ),

        "BgC": (
            BaggingClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 150],
                "max_samples": [0.5, 0.75, 1.0],
                "max_features": [0.5, 0.75, 1.0]
            }
        ),

        "ETC": (
            ExtraTreesClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        ),

        "GBDT": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [2, 3, 5],
                "min_samples_split": [2, 5, 10]
            }
        ),

        "xgb": (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),
            {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        )
    }

    return models


def tune_models(X_train, y_train):

    models = get_tuning_models()

    tuned_models = {}

    for name, (model, param_grid) in models.items():

        print("\n" + "=" * 60)
        print(f"TUNING {name}")
        print("=" * 60)

        # Convert target labels for all models
        y_train_model = y_train.map({
            "No": 0,
            "Yes": 1
        })

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            scoring="f1",
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        search.fit(
            X_train,
            y_train_model
        )

        tuned_models[name] = search

        print(f"\n{name} best parameters:")
        print(search.best_params_)

        print(f"\n{name} best CV F1:")
        print(search.best_score_)

    return tuned_models