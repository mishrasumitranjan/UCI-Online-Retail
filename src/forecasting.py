import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import joblib
import os

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from src.utils import date_column_add_forecasting, date_column_drop, drop_cancelled_orders, drop_non_product_transactions


# Model blending weights for forecasting ensemble
# Used to combine predictions from Ridge regression and Random Forest
WEIGHTS = {
    "ridge": 0.55,
    "rf": 0.45
}

# Maximum values for cyclic encoding of temporal features
# These are used to transform date parts into sine/cosine features
CYCLIC_MAX = {
    "day": 31,          # Days in a month
    "day_of_week": 7,   # Days in a week
    "week": 52,         # Weeks in a year
    "month": 12,        # Months in a year
    "quarter": 4        # Quarters in a year
}


# Define Main Function
def main():
    ...


def clean_data_modeling(df:pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare transactional data for forecasting.

    This function performs a series of preprocessing steps to ensure the
    dataset is suitable for time series forecasting. It removes invalid
    transactions, generates revenue features, and constructs a daily
    aggregated table with lagged and rolling statistics. It also encodes
    date-related features cyclically for use in machine learning models.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactional DataFrame containing at least:
        - 'InvoiceNo' : invoice identifiers.
        - 'StockCode' : product codes.
        - 'Quantity' : number of items purchased.
        - 'UnitPrice' : price per item.
        - 'InvoiceDate' : timestamp of the transaction.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame (`prep_table`) with one row per date, including:
        - 'Revenue' : total daily revenue.
        - Rolling averages and standard deviations (7-day, 30-day).
        - Lagged revenue features (1, 3, 7, 14 days).
        - Cyclically encoded date features (day, day_of_week, week, month, quarter).
        - All non-numeric original date columns dropped.
        - No missing values.

    Notes
    -----
    - Drops duplicate rows, cancelled orders, non-product transactions,
      and inconsistent entries.
    - Ensures only valid transactions (positive quantity and unit price)
      are included.
    - Revenue is aggregated at the daily level to form the forecasting
      target series.
    - Rolling statistics and lag features capture short- and medium-term
      temporal dependencies.
    - Cyclic encoding of date features allows models to learn seasonal
      patterns without discontinuities
    """

    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    # Remove rows with Negative Values and Zero Values
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Dropping cancelled orders
    df = drop_cancelled_orders(df)

    # Dropping non-product transactions
    df = drop_non_product_transactions(df)

    # Remove the two inconsistent entries identified by StockCode and UnitPrice
    df = df[~((df["StockCode"] == "22502") & (df["UnitPrice"] == 649.50))]

    # Generating Revenue column
    df.loc[:, "Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Generating Date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["date"] = df["InvoiceDate"].dt.date

    # Generating prep_table. This table contains the sum of revenue for each date
    prep_table = df.groupby(by=["date"], sort=False).agg({"Revenue": "sum"}).reset_index()

    # Generating additional columns for prep_table.
    prep_table["rolling_7d_avg_revenue"] = prep_table["Revenue"].rolling(7, min_periods=1).mean()
    prep_table["rolling_30d_avg_revenue"] = prep_table["Revenue"].rolling(30, min_periods=1).mean()
    prep_table['rolling_7d_std'] = prep_table['Revenue'].rolling(7, min_periods=1).std()
    prep_table['rolling_30d_std'] = prep_table['Revenue'].rolling(30, min_periods=1).std()

    prep_table["Revenue_lag_1"] = prep_table["Revenue"].shift(1)
    prep_table['Revenue_lag_3'] = prep_table['Revenue'].shift(3)
    prep_table["Revenue_lag_7"] = prep_table["Revenue"].shift(7)
    prep_table['Revenue_lag_14'] = prep_table['Revenue'].shift(14)

    prep_table = date_column_add_forecasting(prep_table)

    for col, max_val in CYCLIC_MAX.items():
        cyclic_encode(prep_table, col, max_val)

    # Drop original columns (keeping only cyclic versions)
    prep_table = date_column_drop(prep_table)

    # Drop rows with missing values
    prep_table.dropna(inplace=True)
    prep_table.reset_index(drop=True, inplace=True)

    return prep_table


# Function to encode cyclic features
def cyclic_encode(data, col, max_val, row_index=None):
    """
    Encode temporal features using cyclic transformation.

    This function applies sine and cosine transformations to a date-related
    feature, converting it into two numeric columns that capture cyclical
    patterns (e.g., days of the week, months of the year). Cyclic encoding
    avoids artificial discontinuities in categorical time variables (e.g.,
    day 31 → day 1) and makes them suitable for machine learning models.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing the column to encode.
    col : str
        Name of the column to encode (e.g., 'day', 'month').
    max_val : int
        Maximum value of the cycle (e.g., 7 for day_of_week, 12 for month).
    row_index : int, optional
        If provided, applies encoding only to the specified row. If None
        (default), applies encoding to the entire column.

    Returns
    -------
    None
        The function modifies the input DataFrame in place by adding two
        new columns:
        - '<col>_sin' : sine transformation
    """

    if row_index is None:
        # Apply to entire column
        data[f"{col}_sin"] = np.sin(2 * np.pi * data[col] / max_val)
        data[f"{col}_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    else:
        # Apply only to one row
        val = data.loc[row_index, col]
        data.loc[row_index, f"{col}_sin"] = np.sin(2 * np.pi * val / max_val)
        data.loc[row_index, f"{col}_cos"] = np.cos(2 * np.pi * val / max_val)


# Function to add a day forecasting
def add_a_day(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend forecasting dataset by adding one future day.

    This function appends a new row representing the next day after the
    latest date in the dataset. It computes rolling statistics, lagged
    features, and cyclic encodings for the new day based on the most recent
    historical data. The resulting DataFrame can be used to generate model
    predictions for the upcoming day.

    Parameters
    ----------
    data_df : pd.DataFrame
        Preprocessed forecasting DataFrame containing at least:
        - 'date' : daily timestamps.
        - 'Revenue' : total daily revenue.
        - Rolling and lagged revenue features.

    Returns
    -------
    pd.DataFrame
        Extended DataFrame including one additional day with:
        - Updated rolling averages and standard deviations (7-day, 30-day).
        - Lagged revenue features (1, 3, 7, 14 days).
        - Cyclically encoded date features for the new day.
        - Validity flags for rolling windows ('rolling_7d_valid',
          'rolling_30d_valid').

    Notes
    -----
    - Uses the last 30 days of data to compute rolling statistics for the
      new day.
    - Lag features are shifted to align with the newly added date.
    - Cyclic encoding ensures temporal features (day, week, month, etc.)
      are represented numerically without discontinuities.
    - Original categorical date columns are dropped, keeping only cyclic
      versions for modelling.
    - Useful for preparing the dataset for one-step-ahead forecasting.
    """

    mini_df = pd.concat(
        [data_df.tail(30),
         pd.DataFrame({"date": [data_df["date"].max() + pd.Timedelta(days=1)]})],
        ignore_index=True
    )

    mini_df = date_column_add_forecasting(mini_df)

    shifted_revenue = mini_df["Revenue"].shift(1)

    mini_df.loc[mini_df.index[-1], "rolling_7d_avg_revenue"] = shifted_revenue.iloc[-8:-1].mean()
    mini_df.loc[mini_df.index[-1], "rolling_30d_avg_revenue"] = shifted_revenue.iloc[-31:-1].mean()
    mini_df.loc[mini_df.index[-1], "rolling_7d_std"] = shifted_revenue.iloc[-8:-1].std()
    mini_df.loc[mini_df.index[-1], "rolling_30d_std"] = shifted_revenue.iloc[-31:-1].std()

    mini_df.loc[mini_df.index[-1], "Revenue_lag_1"] = mini_df["Revenue"].shift(1).iloc[-1]
    mini_df.loc[mini_df.index[-1], "Revenue_lag_3"] = mini_df["Revenue"].shift(3).iloc[-1]
    mini_df.loc[mini_df.index[-1], "Revenue_lag_7"] = mini_df["Revenue"].shift(7).iloc[-1]
    mini_df.loc[mini_df.index[-1], "Revenue_lag_14"] = mini_df["Revenue"].shift(14).iloc[-1]

    mini_df.loc[mini_df.index[-1], "rolling_7d_valid"] = mini_df.index[-1] >= 6
    mini_df.loc[mini_df.index[-1], "rolling_30d_valid"] = mini_df.index[-1] >= 29

    last_idx = mini_df.index[-1]
    for col, max_val in CYCLIC_MAX.items():
        cyclic_encode(mini_df, col, max_val, row_index=last_idx)

    mini_df = date_column_drop(mini_df)

    return mini_df


class RevenueForecastPipeline:
    """
    RevenueForecastPipeline

    A machine learning pipeline for forecasting daily revenue using an
    ensemble of Ridge regression and Random Forest models. The pipeline
    handles data preprocessing, feature engineering, model training,
    prediction, evaluation, visualisation, and saving/loading of trained
    models. It also supports multi-day forecasting by iteratively extending
    the dataset with engineered features.

    Attributes
    ----------
    scaler : StandardScaler
        Scaler used to normalise feature values.
    tscv : TimeSeriesSplit
        Cross-validation splitter for time series data.
    ridge : Ridge
        Ridge regression model for forecasting.
    rf : RandomForestRegressor
        Random Forest model for forecasting.
    weights : dict
        Blending weights for combining Ridge and Random Forest predictions.
    features : list
        List of feature column names used for training.
    target : str
        Name of the target variable ('Revenue').
    prep_table : pd.DataFrame
        Preprocessed dataset with engineered features.
    X_train, X_test : np.ndarray
        Training and test feature sets.
    y_train, y_test : np.ndarray
        Training and test target values.
    pred_ridge, pred_rf, blend : np.ndarray
        Predictions from Ridge, Random Forest, and blended ensemble.
    forecasted_table: pd.DataFrame
        Dataset containing forecasted data.

    Methods
    -------
    preprocess(df):
        Clean and preprocess raw transactional data, define features and target.
    fit():
        Train Ridge and Random Forest models using time series cross-validation.
    predict(X_test=None):
        Generate predictions from Ridge, Random Forest, and blended ensemble.
    evaluate(y_true=None, predictions=None):
        Evaluate model performance using MAPE, MSE, and R² metrics.
    graph_blended_results():
        Visualise baseline, Ridge, Random Forest, and blended predictions.
    save(path="models"):
        Save trained models and scaler to disk.
    load(path="models"):
        Load trained models and scaler from disk.
    forecast_next_day(day_features):
        Forecast revenue for the next day using blended predictions.
    forecast(prediction_days=7):
        Forecast revenue for multiple future days by iteratively extending
        the dataset.
    """

    def __init__(self):
        """
        Initialise the RevenueForecastPipeline.

        This constructor sets up all core components required for the
        forecasting workflow, including preprocessing tools, regression
        models, blending weights, and placeholders for data and results.

        Attributes
        ----------
        scaler : StandardScaler
            Scales feature values to zero mean and unit variance.
        tscv : TimeSeriesSplit
            Cross-validation splitter for sequential time series data.
        ridge : Ridge
            Ridge regression model with alpha=0.5 and max_iter=10000.
        rf : RandomForestRegressor
            Random Forest model with 200 trees, max depth of 6, and fixed
            random state for reproducibility.
        weights : dict
            Blending weights for combining Ridge and Random Forest predictions.
        features : list or None
            List of feature column names, defined during preprocessing.
        target : str or None
            Name of the target variable (Revenue).
        prep_table : pd.DataFrame or None
            Preprocessed dataset used for training and forecasting.
        X_train, X_test : np.ndarray or None
            Training and test feature matrices.
        y_train, y_test : np.ndarray or None
            Training and test target arrays.
        pred_ridge, pred_rf, blend : np.ndarray or None
            Stored predictions from Ridge, Random Forest, and blended ensemble.
        forecasted_table : pd.DataFrame or None
            Forecasted revenue table generated by the forecast method.
        """
        self.scaler = StandardScaler()
        self.tscv = TimeSeriesSplit(n_splits=5)
        self.ridge = Ridge(alpha=0.5, max_iter=10000)
        self.rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=2315, n_jobs=-1)
        self.weights = WEIGHTS
        self.features, self.target = None, None
        self.prep_table = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pred_ridge = None
        self.pred_rf = None
        self.blend = None
        self.forecasted_table = None

    def preprocess(self, df: pd.DataFrame) -> None:
        """
        Clean and preprocess the input dataset for forecasting.

        This method applies the `clean_data_modeling` function to the raw
        DataFrame, producing a preprocessed table suitable for model training
        and forecasting. It also defines the feature set and target variable
        used throughout the pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame containing at least:
            - 'date' : daily timestamps.
            - 'Revenue' : target variable for forecasting.
            - Additional explanatory features.

        Returns
        -------
        None
            Updates internal attributes:
            - self.prep_table : pd.DataFrame
                Cleaned and preprocessed dataset.
            - self.features : list[str]
                Names of feature columns (all except 'Revenue' and 'date').
            - self.target : str
                Name of the target variable ('Revenue').
        """
        # Clean and preprocess the data
        self.prep_table = clean_data_modeling(df.copy())

        # Define features and target variable
        # self.features = [col for col in self.prep_table.columns if col != "Revenue" and col != "date"]
        self.features = [col for col in self.prep_table.columns if col not in ("Revenue", "date")]
        self.target = "Revenue"

    def _scale_data(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the scaler on training data and transform both training and test sets.

        This internal helper method ensures that the scaling parameters
        (mean and variance) are derived only from the training set, then
        applied consistently to both training and test features.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix before scaling.
        X_test : np.ndarray
            Test feature matrix before scaling.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            - Scaled training feature matrix.
            - Scaled test feature matrix.
        """
        self.scaler.fit(X_train)
        return self.scaler.transform(X_train), self.scaler.transform(X_test)

    def fit(self) -> None:
        """
        Train the forecasting models using time series cross-validation.

        This method splits the preprocessed dataset into sequential
        training and test sets using `TimeSeriesSplit`. For each split,
        it scales the features, flattens the target arrays, and fits
        both the Random Forest and Ridge regression models. The most
        recent split is stored in the pipeline attributes for evaluation
        and prediction. Raises ValueError if prep_table
        is not available.

        Updates
        -------
        self.X_train : np.ndarray
            Scaled training feature matrix from the latest split.
        self.X_test : np.ndarray
            Scaled test feature matrix from the latest split.
        self.y_train : np.ndarray
            Training target values from the latest split.
        self.y_test : np.ndarray
            Test target values from the latest split.
        self.rf : RandomForestRegressor
            Fitted Random Forest model.
        self.ridge : Ridge
            Fitted Ridge regression model.

        Returns
        -------
        None
        """
        if self.prep_table is None:
            raise ValueError("Preprocessing must be performed before fitting. Call preprocess(df).")

        for train_index, test_index in self.tscv.split(self.prep_table):
            train, test = self.prep_table.iloc[train_index], self.prep_table.iloc[test_index]

            X_train, X_test = self._scale_data(train[self.features], test[self.features])
            self.X_train, self.X_test = X_train, X_test

            y_train, y_test = train[[self.target]], test[[self.target]]

            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
            self.y_train, self.y_test = y_train, y_test

            self.rf.fit(X_train, y_train)
            self.ridge.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions using the trained models.

        This method produces forecasts from both the Ridge regression
        and Random Forest models, then combines them into a blended
        prediction using the predefined weights. If no test set is
        provided, it defaults to the most recent split stored in
        `self.X_test`. Falls back to calling fit() if training has
        not yet been performed.

        Parameters
        ----------
        X_test : np.ndarray, optional
            Feature matrix to predict on. If None, uses `self.X_test`
            from the latest training split.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            - Ridge regression predictions.
            - Random Forest predictions.
            - Blended ensemble predictions.
        """
        if self.X_test is None:
            self.fit()

        if X_test is None:
            X_test = self.X_test

        self.pred_ridge = self.ridge.predict(X_test)
        self.pred_rf = self.rf.predict(X_test)
        self.blend = self.weights["ridge"] * self.pred_ridge + self.weights["rf"] * self.pred_rf
        return self.pred_ridge, self.pred_rf, self.blend

    def evaluate(
            self,
            y_true: np.ndarray | None = None,
            predictions: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Evaluate model performance on test data.

        This method computes key regression metrics (MAPE, MSE, R²) to
        assess the accuracy of the blended forecast. If no true values
        or predictions are provided, it defaults to using `self.y_test`
        and the blended predictions from `self.predict()`. Falls back
        to calling predict() if prediction has not yet been performed.

        Parameters
        ----------
        y_true : np.ndarray, optional
            Ground truth target values. Defaults to `self.y_test`.
        predictions : np.ndarray, optional
            Predicted values to evaluate. Defaults to the blended
            ensemble predictions (`self.blend`).

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - "MAPE" : Mean Absolute Percentage Error.
            - "MSE"  : Mean Squared Error.
            - "R2"   : Coefficient of determination (R² score).
        """
        if self.blend is None:
            self.predict()

        # Calculate and compare the predictions.
        if any(p is None for p in [self.pred_ridge, self.pred_rf, self.blend]):
            self.predict()

        y_true = self.y_test if y_true is None else y_true
        predictions = self.blend if predictions is None else predictions

        # Warning: Expected type 'dict[str, float]', got 'dict[str, float | array | Any]' instead
        # return {
        #     "MAPE": mean_absolute_percentage_error(y_true, predictions),
        #     "MSE": mean_squared_error(y_true, predictions),
        #     "R2": r2_score(y_true, predictions)
        # }

        return {
            "MAPE": float(np.squeeze(mean_absolute_percentage_error(y_true, predictions))),
            "MSE": float(np.squeeze(mean_squared_error(y_true, predictions))),
            "R2": float(np.squeeze(r2_score(y_true, predictions)))
        }

    # def graph_blended_results(self): #DEPRECATED
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     sns.lineplot(x=range(len(self.y_test)), y=self.y_test, label="Baseline", ax=ax)
    #     sns.lineplot(x=range(len(self.pred_ridge)), y=self.pred_ridge, color="red", label="Ridge", alpha=0.7, ax=ax)
    #     sns.lineplot(x=range(len(self.pred_rf)), y=self.pred_rf, color="black", label="Random Forest", alpha=0.7, ax=ax)
    #     sns.lineplot(x=range(len(self.blend)), y=self.blend, color="green", label="Best Blend", alpha=0.7, ax=ax)
    #     return fig

    def graph_blended_results(self) -> go.Figure:
        """
        Visualise blended model results using Plotly.

        Plots baseline (true values), Ridge predictions, Random Forest
        predictions, and blended ensemble predictions for comparison.
        Falls back to calling predict() if prediction has not yet
        been performed.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive line chart comparing actual vs predicted values.
        """
        if self.blend is None:
            self.predict()

        fig = go.Figure()

        # Baseline (true values)
        fig.add_trace(go.Scatter(
            x=list(range(len(self.y_test))),
            y=self.y_test,
            mode="lines",
            name="Baseline",
            line=dict(color="blue")
        ))

        # Ridge predictions
        fig.add_trace(go.Scatter(
            x=list(range(len(self.pred_ridge))),
            y=self.pred_ridge,
            mode="lines",
            name="Ridge",
            line=dict(color="red", dash="dash")
        ))

        # Random Forest predictions
        fig.add_trace(go.Scatter(
            x=list(range(len(self.pred_rf))),
            y=self.pred_rf,
            mode="lines",
            name="Random Forest",
            line=dict(color="yellow", dash="dot")
        ))

        # Blended predictions
        fig.add_trace(go.Scatter(
            x=list(range(len(self.blend))),
            y=self.blend,
            mode="lines",
            name="Best Blend",
            line=dict(color="green")
        ))

        fig.update_layout(
            title="Blended Forecast Results",
            xaxis_title="Time Index",
            yaxis_title="Revenue",
            width=1080,
            height=500
        )

        return fig

    def save(self, path: str = "models") -> None:
        """
        Save trained models and preprocessing objects to disk.

        This method ensures the target directory exists, then serializes
        the Ridge regression model, Random Forest model, and the fitted
        scaler using joblib. The files are stored with fixed names for
        consistency and later reuse.

        Parameters
        ----------
        path : str, default="models"
            Directory path where the models and scaler will be saved.
            If the folder does not exist, it will be created.

        Returns
        -------
        None
            Models and scaler are written to disk as:
            - "ridge_model.pkl"
            - "rf_model_2315.pkl"
            - "scaler.pkl"
        """
        # Ensure the folder exists
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.ridge, os.path.join(path, "ridge_model.pkl"))
        joblib.dump(self.rf, os.path.join(path, "rf_model_2315.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))

    def load(self, path: str = "models") -> None:
        """
        Load trained models and preprocessing objects from disk.

        This method restores the Ridge regression model, Random Forest
        model, and the fitted scaler from serialized joblib files in
        the specified directory. It assumes the files were previously
        saved using the `save` method.

        Parameters
        ----------
        path : str, default="models"
            Directory path where the models and scaler are stored.

        Returns
        -------
        None
            Updates internal attributes:
            - self.ridge : Ridge
                Loaded Ridge regression model.
            - self.rf : RandomForestRegressor
                Loaded Random Forest model.
            - self.scaler : StandardScaler
                Loaded fitted scaler.
        """
        self.ridge = joblib.load(os.path.join(path, "ridge_model.pkl"))
        self.rf = joblib.load(os.path.join(path, "rf_model_2315.pkl"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))

    def forecast_next_day(self, day_features: pd.DataFrame) -> float:
        """
        Forecast revenue for the next day.

        This method takes a single day's feature set, scales it using the
        fitted scaler, generates predictions from both the Ridge regression
        and Random Forest models, and blends them using the predefined
        weights. The blended forecast is returned as a scalar float. Falls
        back to calling fit() if training has not yet been performed.

        Parameters
        ----------
        day_features : pd.DataFrame
            A DataFrame containing the feature values for the day to be
            forecasted. Must include all columns listed in `self.features`.

        Returns
        -------
        float
            Blended revenue forecast for the next day.
        """
        if self.X_test is None:
            self.fit()

        # Prepare input
        forecast_features = self.scaler.transform(day_features[self.features])

        # Predict
        ridge_forecast = self.ridge.predict(forecast_features)
        rf_forecast = self.rf.predict(forecast_features)
        blend_forecast = self.weights["ridge"] * ridge_forecast + self.weights["rf"] * rf_forecast

        return blend_forecast.ravel()[0]

    def forecast(self, prediction_days: int = 7) -> pd.DataFrame:
        """
        Generate multi-day revenue forecasts.

        This method iteratively extends the preprocessed dataset by adding
        new days, forecasting each day's revenue using the blended model,
        and appending the results to a forecast table. By default, it
        produces a 7-day forecast horizon.

        Parameters
        ----------
        prediction_days : int, default=7
            Number of future days to forecast.

        Returns
        -------
        pd.DataFrame
            Dataset containing forecasted dates and revenues with columns:
            - "date" : forecasted date.
            - "Revenue" : blended forecasted revenue.
        """
        forecasted_table = pd.DataFrame(columns=["date", self.target])
        mini_df = self.prep_table.tail(30).copy()

        for d in range(prediction_days):
            mini_df = add_a_day(mini_df)
            mini_df.loc[mini_df.index[-1], self.target] = self.forecast_next_day(mini_df.tail(1))

            # append to forecasted_table
            forecasted_table = pd.concat(
                [forecasted_table, mini_df.tail(1)[["date", self.target]]],
                ignore_index=True
            )

        self.forecasted_table = forecasted_table

        return forecasted_table

    def graph_forecasted_results(self, history_days: int = 30, forecast_days: int = 7) -> go.Figure:
        """
        Visualize historical and forecasted revenue trends.

        This method generates an interactive Plotly line chart comparing
        recent historical revenue values with forecasted results. If no
        forecast has been generated yet, it automatically calls `forecast()`
        with default arguments to produce one.

        Parameters
        ----------
        history_days : int, default=30
            Number of past days to display from the preprocessed dataset.
            The value is clipped to the available length of `self.prep_table`.
        forecast_days : int, default=7
            Number of forecasted days to display from the forecasted dataset.
            The value is clipped to the available length of `self.forecasted_table`.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive line chart with two traces:
            - "Historical" : actual revenue values from the past `history_days`.
            - "Forecast"   : blended forecasted revenue values for the next `forecast_days`.
        """
        if self.forecasted_table is None:
            self.forecast()

        # To ensure numbers are non-negative and within bounds
        history_days = max(1, min(history_days, len(self.prep_table)))
        forecast_days = max(1, min(forecast_days, len(self.forecasted_table)))

        fig = go.Figure()

        # Historical slice
        hist_df = self.prep_table.tail(history_days)
        fig.add_trace(go.Scatter(
            x=hist_df["date"],
            y=hist_df["Revenue"],
            mode="lines",
            name="Historical",
            line=dict(color="blue")
        ))

        # Forecast slice
        forecast_df = self.forecasted_table.head(forecast_days)
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["Revenue"],
            mode="lines",
            name="Forecast",
            line=dict(color="red", dash="dash")
        ))

        fig.update_layout(
            title="Historical vs Forecasted Revenue",
            xaxis_title="Date",
            yaxis_title="Revenue",
            width=1080,
            height=500
        )

        return fig

    def run_pipeline(self, df: pd.DataFrame, prediction_days: int = 7) -> pd.DataFrame:
        """
        Execute the full forecasting pipeline in the intended order.

        This method provides a high-level entry point for users who want
        to run the complete workflow without calling individual methods.
        It cleans and preprocesses the data, fits the models, and
        produces a forecasted table for the specified number of future
        days.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transactional dataset containing 'InvoiceNo', 'StockCode',
            'Quantity', 'UnitPrice', and 'InvoiceDate'.
        prediction_days : int, default=7
            Number of future days to forecast.

        Returns
        -------
        pd.DataFrame
            Forecasted table with dates and predicted revenue for the
            specified horizon.
        """

        self.preprocess(df)

        self.fit()

        forecasted_table = self.forecast(prediction_days)
        self.forecasted_table = forecasted_table

        return forecasted_table

if __name__ == "__main__":
    main()