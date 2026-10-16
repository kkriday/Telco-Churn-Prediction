"""Loading and cleaning the IBM Telco Customer Churn dataset."""
import pandas as pd

from .config import DATA_PATH, ID_COL, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and the TotalCharges gap.

    TotalCharges is stored as text. Its 11 blank values all belong to customers
    with tenure 0 who have not been billed yet, so the correct value is 0,
    not an imputed median.
    """
    df = df.copy()
    blanks = df["TotalCharges"].str.strip() == ""
    assert (df.loc[blanks, "tenure"] == 0).all(), "blank TotalCharges outside tenure 0"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", "0"))
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET, ID_COL]), df[TARGET]
