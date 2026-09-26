import pandas as pd


DROP_COLUMNS = [
    "EmployeeCount",
    "StandardHours",
    "Over18",
    "EmployeeNumber"
]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the IBM HR Attrition dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw HR dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    df = df.copy()

    # Remove constant and identifier columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df
