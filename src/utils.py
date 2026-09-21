import numpy as np
import pandas as pd
import calendar
import os
from pathlib import Path

def main():
    ...


# Function to load the data
def load_data(path: str|Path, fast_format: str = "csv") -> pd.DataFrame:
    """
    Load a dataset from any pandas-supported format, with fallback to a faster format.

    If the file is already in the chosen `fast_format` (default: CSV), it is read directly.
    If the file is in another format (e.g., XLSX, JSON, Parquet), the function checks
    whether a converted file with the same base name and `fast_format` extension exists:
        - If found, loads the converted file.
        - If not found, reads the original file, saves it in the faster format, and then
          loads the converted file.

    Parameters
    ----------
    path : str
        Path to the dataset in any pandas-supported format.
    fast_format : str, default="csv"
        Target format to normalise to for faster future reads.
        Options include: "csv", "parquet", "feather", etc.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """

    # Normalise extension
    ext = Path(path).suffix.lower()
    fast_path = Path(path).with_suffix(f".{fast_format}")

    expected_columns = [
        'InvoiceNo', 'StockCode', 'Description', 'Quantity',
        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
    ]
    # default_path = os.path.join(os.path.dirname(__file__), "..", "data", "Online Retail.csv") # type: ignore
    default_path = Path(__file__).resolve().parent.parent / "data" / "Online Retail.csv"

    # Case 1: Already in fast format
    if ext == f".{fast_format}":
        if fast_format == "csv":
            return pd.read_csv(path)
        elif fast_format == "parquet":
            return pd.read_parquet(path)
        elif fast_format == "feather":
            return pd.read_feather(path)
        else:
            print(f"Unsupported fast_format: {fast_format}")
            

    # Case 2: Converted file exists
    if os.path.exists(fast_path):
        if fast_format == "csv":
            return pd.read_csv(fast_path)
        elif fast_format == "parquet":
            return pd.read_parquet(fast_path)
        elif fast_format == "feather":
            return pd.read_feather(fast_path)

    # Case 3: Read original, convert, then reload
    df = pd.read_excel(path) if ext in [".xls", ".xlsx"]\
        else pd.read_table(path) if ext in [".txt"]\
        else pd.read_json(path) if ext == ".json"\
        else pd.read_parquet(path) if ext == ".parquet"\
        else pd.read_feather(path) if ext == ".feather"\
        else pd.read_csv(path)

    # Validate Schema
    if not all(col in df.columns for col in expected_columns):
        print("Invalid dataset: missing required columns. Falling back to default dataset.")
        df = pd.read_csv(default_path)

    # Save to fast format
    if fast_format == "csv":
        df.to_csv(fast_path, index=False)
        return pd.read_csv(fast_path)
    elif fast_format == "parquet":
        df.to_parquet(fast_path, index=False)
        return pd.read_parquet(fast_path)
    elif fast_format == "feather":
        df.to_feather(fast_path)
        return pd.read_feather(fast_path)
    else:
        raise ValueError(f"Unsupported fast_format: {fast_format}")


# Function to add date data columns to a dataframe.
def _add_common_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common date-related columns to a DataFrame.

    This helper function converts the 'date' column to a proper datetime
    format and enriches the DataFrame with additional time-based features
    (day, week, quarter). It standardises date handling for downstream
    analysis and ensures consistent date parsing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following added columns:
        - 'date' : converted to datetime (day-first format).
        - 'day' : day of the month.
        - 'week' : ISO calendar week number.
        - 'quarter' : quarter of the year.

    Notes
    -----
    - The 'date' column is parsed with `dayfirst=True` to handle day-first
      formats correctly.
    - Useful for exploratory data analysis, time-based grouping, and
      seasonal trend detection.
    """

    # df.loc[:, "date"] = pd.to_datetime(df["date"], dayfirst=True)
    # df.loc[:, "day"] = df["date"].dt.day
    # df.loc[:, "week"] = df["date"].dt.isocalendar().week
    # df.loc[:, "quarter"] = df["date"].dt.quarter
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    return df


def date_column_add_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extended date-related columns for forecasting.

    This function enriches a DataFrame with additional time-based features
    to support forecasting and time series analysis. It builds on the
    common date parts (day, week, quarter) and adds day-of-week and month
    columns for more granular temporal patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following added columns:
        - 'date' : converted to datetime (day-first format).
        - 'day' : day of the month.
        - 'week' : ISO calendar week number.
        - 'quarter' : quarter of the year.
        - 'day_of_week' : day of the week (0 = Monday, 6 = Sunday).
        - 'month' : month of the year (1–12).

    Notes
    -----
    - Builds on `_add_common_date_parts` to ensure consistent date parsing.
    - Useful for forecasting models, seasonal decomposition, and identifying
      weekly or monthly trends.
    """

    df = _add_common_date_parts(df)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def date_column_add_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extended date-related columns for exploratory data analysis (EDA).

    This function enriches a DataFrame with detailed time-based features
    derived from the 'InvoiceDate' column. It builds on the common date
    parts (day, week, quarter) and adds categorical representations of
    day-of-week and month, as well as year, time, and hour. These features
    are useful for identifying temporal patterns in customer behaviour and
    transaction activity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing an 'InvoiceDate' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following added columns:
        - 'date' : converted to datetime (day-first format).
        - 'day' : day of the month.
        - 'week' : ISO calendar week number.
        - 'quarter' : quarter of the year.
        - 'day_of_week' : categorical day name (Monday–Sunday).
        - 'month' : categorical month name (January–December).
        - 'year' : year of the invoice date.
        - 'time' : time component of the invoice date.
        - 'hour' : hour of the invoice date.

    Notes
    -----
    - Builds on `_add_common_date_parts` to ensure consistent date parsing.
    - Day-of-week and month are stored as ordered categorical variables,
      preserving natural calendar order.
    - Useful for EDA tasks such as analysing seasonality, weekly trends,
      and hourly purchasing behaviour.
    """

    df = _add_common_date_parts(df)
    # df.loc[:, "day_of_week"] = pd.Categorical(df["InvoiceDate"].dt.day_name(), list(calendar.day_name), ordered=True)
    # df.loc[:, "month"] = pd.Categorical(df["InvoiceDate"].dt.month_name(), list(calendar.month_name)[1:], ordered=True)
    # df.loc[:, "year"] = df["InvoiceDate"].dt.year
    # df.loc[:, "time"] = df["InvoiceDate"].dt.time
    # df.loc[:, "hour"] = df["InvoiceDate"].dt.hour
    df["day_of_week"] = pd.Categorical(df["InvoiceDate"].dt.day_name(), list(calendar.day_name), ordered=True)
    df["month"] = pd.Categorical(df["InvoiceDate"].dt.month_name(), list(calendar.month_name)[1:], ordered=True)
    df["year"] = df["InvoiceDate"].dt.year
    df["time"] = df["InvoiceDate"].dt.time
    df["hour"] = df["InvoiceDate"].dt.hour
    return df


# Corresponding function to drop date data columns from a dataframe.
def date_column_drop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop original date-derived columns after cyclic encoding.

    This function removes non-numeric date-related features (day, day_of_week,
    week, month, quarter) from a DataFrame, keeping only their cyclic
    representations. It ensures the dataset is ready for forecasting or
    machine learning models that require numeric inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing both original date-derived columns and
        their cyclic-encoded counterparts.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original categorical date-related columns removed,
        leaving only cyclic (numeric) versions for modelling.

    Notes
    -----
    - Intended as a clean-up step after feature engineering with cyclic
      encodings of temporal variables.
    - Helps avoid redundancy and ensures compatibility with models that
      cannot directly handle categorical date features.
    - Complements `date_column_add_forecasting` or similar functions that
      generate cyclic encodings for time-based features.
    """

    df.drop(columns=["day", "day_of_week", "week", "month", "quarter"], inplace=True)
    return df


def drop_cancelled_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelled orders from a DataFrame.

    This function filters out cancelled transactions based on the invoice
    number convention. In many retail datasets, cancelled orders are marked
    with an 'InvoiceNo' starting with "C". By removing these rows, the
    resulting DataFrame contains only valid (non-cancelled) transactions,
    making it suitable for analysis and forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing an 'InvoiceNo' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with cancelled orders removed.

    Notes
    -----
    - Cancelled orders are identified by 'InvoiceNo' values that start with "C".
    - Ensures that downstream analysis (e.g., revenue calculations, customer
      segmentation, forecasting) is based only on completed transactions.
    - Useful as a preprocessing step in data cleaning pipelines.
    """

    mask = df["InvoiceNo"].str.startswith("C", na=False)
    return df[~mask]


def drop_non_product_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove non-product transactions from a DataFrame.

    This function filters out rows associated with non-product stock codes.
    In many retail datasets, certain `StockCode` values represent shipping,
    discounts, or administrative entries rather than actual products. These
    codes are typically non-numeric. By removing them, the resulting DataFrame
    contains only genuine product transactions suitable for analysis and
    forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'StockCode' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with non-product transactions removed.

    Notes
    -----
    - Non-product codes are identified as `StockCode` values containing only
      non-digit characters.
    - Two specific codes (`DCGSSBOY`, `DCGSSGIRL`) are exceptions and retained,
      even though they match the non-digit pattern.
    - Ensures that downstream analysis (e.g., demand forecasting, product-level
      revenue analysis) is based only on valid product transactions.
    - Useful as a preprocessing step in data cleaning pipelines to remove
      irrelevant entries.
    """
    non_product_codes = df[df["StockCode"].str.contains(r"^\D+$", na=False)]["StockCode"].unique()
    to_remove = ["DCGSSBOY", "DCGSSGIRL"]  # Two products found to be having non-product codes
    indices_to_remove = np.where(np.isin(non_product_codes, to_remove))
    non_product_codes = np.delete(non_product_codes, indices_to_remove)
    return df[~df["StockCode"].isin(non_product_codes)]


if __name__ == "__main__":
    main()
