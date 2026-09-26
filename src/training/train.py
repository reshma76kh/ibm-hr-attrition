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


def get_models():

    models = {
        "SVC": SVC(
            kernel="sigmoid",
            gamma=1.0
        ),

        "KN": KNeighborsClassifier(),

        "DT": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        "LR": LogisticRegression(
            solver="liblinear",
            penalty="l1",
            random_state=42
        ),

        "RF": RandomForestClassifier(
            n_estimators=50,
            random_state=42
        ),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        ),

        "BgC": BaggingClassifier(
            n_estimators=50,
            random_state=42
        ),

        "ETC": ExtraTreesClassifier(
            n_estimators=50,
            random_state=42
        ),

        "GBDT": GradientBoostingClassifier(
            n_estimators=50,
            random_state=42
        ),

        "xgb": XGBClassifier(
            n_estimators=50,
            random_state=42,
            eval_metric="logloss"
        )
    }

    return models


# REPLACE your old train_models() with this

def train_models(X_train, y_train):

    models = get_models()

    trained_models = {}

    for name, model in models.items():

        print(f"Training {name}...")

        if name == "xgb":

            y_train_xgb = y_train.map({
                "No": 0,
                "Yes": 1
            })

            model.fit(
                X_train,
                y_train_xgb
            )

        else:

            model.fit(
                X_train,
                y_train
            )

        trained_models[name] = model

        print(f"{name} trained successfully.")

    return trained_models