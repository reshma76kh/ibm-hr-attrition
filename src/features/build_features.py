import pandas as pd


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "Attrition"
):
    """
    Separate the target variable from the input features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned HR dataset.

    target_column : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Input features.

    y : pd.Series
        Target variable.
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y