import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from src.utils import date_column_add_eda, date_column_drop, drop_cancelled_orders, drop_non_product_transactions

"""
Pattern: <aggregation>_<column>_by_<group>
E.g:
sum_revenue_by_country
mean_quantity_by_customer
unique_invoice_by_country
"""


# Define Main Function
def main():
    ...


# Define Additional Functions
def clean_data_eda(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and preprocess transactional data for exploratory data analysis (EDA).

    This function performs a series of data cleaning and transformation steps to
    prepare the dataset for analysis. It handles duplicates, missing values,
    inconsistent entries, type conversions, and generates derived columns for
    revenue and temporal features. It also builds a dictionary mapping each
    StockCode to its most frequent Description, which can be reused across
    analyses to avoid repeated computation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing transactional data with at least the following columns:
        - 'InvoiceNo' : invoice identifiers.
        - 'StockCode' : product codes.
        - 'Description' : product descriptions.
        - 'Quantity' : number of items purchased.
        - 'InvoiceDate' : timestamp of the transaction.
        - 'UnitPrice' : price per unit.
        - 'CustomerID' : customer identifiers.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple containing:
        - Cleaned and enriched DataFrame with:
            - Duplicates removed.
            - Missing 'CustomerID' values replaced with 0 and cast to int.
            - 'StockCode' cast to string.
            - 'Description' standardised (trimmed, lowercased, filled using mode per StockCode, and title-cased).
            - Negative or zero 'Quantity' and 'UnitPrice' rows removed.
            - Cancelled orders dropped.
            - Non-product transactions dropped.
            - Specific inconsistent entries removed (StockCode "22502" with UnitPrice 649.50).
            - 'InvoiceNo' cast to int.
            - 'Revenue' column generated as Quantity × UnitPrice.
            - 'date' column generated from 'InvoiceDate'.
            - Additional temporal columns added: 'year', 'time', 'hour', plus those from `date_column_add`.
            - Outliers in 'Quantity' flagged in 'is_outlier' using IQR method (outer bound = Q3 + 3 × IQR).
        - Dictionary mapping each StockCode to its most frequent Description
          (entries with insufficient data or no mode are excluded).

    Notes
    -----
    - Outliers are flagged but not removed.
    - Helper functions `drop_cancelled_orders`, `drop_non_product_transactions`,
      and `date_column_add` must be defined elsewhere.
    - The IQR multiplier is set to 3 to preserve more data points; can be adjusted to 1.5 if desired.
    - The description dictionary is built once during cleaning and can be reused
      across analyses to reduce processing time.
    """
    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    # Replacing NA CustomerIDs with 0.
    df["CustomerID"].fillna(0, inplace=True)

    # Forced Type Casting to avoid type-related errors
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str)

    # Trimming and standardising descriptions for easier processing.
    df.loc[:, "Description"] = df["Description"].str.strip()
    df.loc[:, "Description"] = df["Description"].str.lower()

    # Get all the entries with Description as NaN.
    desc_na = df[df["Description"].isna()]

    # Convert to dictionary
    description_mode = build_description_mode(df)

    # Fill NaN values in Description column where possible.
    for idx in desc_na.index:
        try:
            df.loc[idx, "Description"] = description_mode[df.loc[idx, "StockCode"]]
        except KeyError:
            pass

    # Make the descriptions more presentable
    for k, v in description_mode.items():
        description_mode[k] = v.title()

    # Filter out the rows with negative values and zero values for Quantity and Unit Price.
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Dropping cancelled orders
    df = drop_cancelled_orders(df)

    # Dropping non-product transactions
    df = drop_non_product_transactions(df)

    # Remove the two inconsistent entries identified by StockCode and UnitPrice
    df = df[~((df["StockCode"] == "22502") & (df["UnitPrice"] == 649.50))]

    # With only numerical values left, the column's datatype can be changed to 'int'.
    df["InvoiceNo"] = df["InvoiceNo"].astype(int)

    # Generating Revenue column
    df.loc[:, "Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Generating Date column
    df["date"] = df["InvoiceDate"].dt.date

    # Generating additional date and time related columns
    df = date_column_add_eda(df)

    # Identifying outliers based on Quantity.
    qty_details = df["Quantity"].describe()
    qty_IQR = qty_details["75%"] - qty_details["25%"]

    # 3 was used in the IQR method to identify outliers to preserve more data points.
    qty_outer_bound = qty_details["75%"] + (3 * qty_IQR)

    # Outliers are marked in a separate column instead of removing them.
    df["is_outlier"] = df["Quantity"] > qty_outer_bound

    # Uncomment to use 1.5 instead of 3.
    # qty_inner_bound = qty_details["75%"] + (1.5 * qty_IQR)
    # df["is_outlier"] = df["Quantity"] > qty_inner_bound

    return df, description_mode


def build_description_mode(df: pd.DataFrame) -> dict:
    """
    Build a dictionary mapping each StockCode to its most frequent Description.

    The function groups the DataFrame by StockCode and determines the mode
    (most common value) of the Description column for each product. If a product
    has fewer than 3 rows or no valid mode, it is excluded from the result.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifier.
        - 'Description' : product description (may be inconsistent).

    Returns
    -------
    dict
        Dictionary with StockCode as keys and most frequent Description as values.
        Entries with insufficient data or no mode are excluded.
    """

    def most_frequent_desc(x):
        if len(x) <= 2:
            return None
        mode_vals = x.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else None

    description_mode_series = df.groupby("StockCode")["Description"].agg(most_frequent_desc)
    return description_mode_series.dropna().to_dict()


# Graph-related functions
def apply_common_layout(
        fig: go.Figure,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        width: int = 1080,
        height: int = 500,
        showlegend: bool = True,
        legend_position: tuple = (0.02, 0.96),
        grids: str | None = "both"
) -> go.Figure:
    """
    Apply a standardised layout to a Plotly figure.

    This function updates a Plotly figure with consistent styling, including
    titles, axis labels, dimensions, gridline visibility, and legend
    placement. It ensures a uniform look across visualisations while
    allowing flexible customisation.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to update.
    title : str
        Title of the plot.
    xaxis_title : str
        Label for the x-axis.
    yaxis_title : str
        Label for the y-axis.
    width : int, optional
        Width of the figure in pixels (default: 1080).
    height : int, optional
        Height of the figure in pixels (default: 500).
    showlegend : bool, optional
        Whether to display the legend (default: True).
    legend_position : tuple, optional
        (x, y) coordinates for legend placement (default: (0.02, 0.96)).
    grids : str, optional
        Gridline visibility setting: None, "x", "y", or "both" (default: "both").

    Returns
    -------
    go.Figure
        Updated Plotly figure with standardised layout applied.

    Notes
    -----
    - Gridline visibility is controlled via the `grids` parameter.
    - Legend styling includes semi-transparent background and border.
    - Layout uses the "plotly_white" template for a clean appearance.
    - Ensures consistent sizing and axis formatting across figures.
    """

    if grids is not None:
        grids = grids.lower()
        if grids not in {"x", "y", "both", "none"}:
            raise ValueError("grids must be one of: None, 'x', 'y', 'both', or 'none'")

    show_xgrid = grids in ("x", "both")
    show_ygrid = grids in ("y", "both")

    layout_kwargs = dict(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        xaxis=dict(showgrid=show_xgrid, gridcolor="rgba(211, 211, 211, 0.5)", gridwidth=0.5),
        yaxis=dict(showgrid=show_ygrid, gridcolor="rgba(211, 211, 211, 0.5)", gridwidth=0.5),
        width=width,
        height=height
    )

    if showlegend:
        layout_kwargs["legend"] = dict(
            x=legend_position[0], y=legend_position[1],
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="black", borderwidth=0.5
        )
    else:
        layout_kwargs["showlegend"] = False

    fig.update_layout(**layout_kwargs)
    return fig


def make_line_trace(
    x: pd.Series,
    y: pd.Series,
    name: str,
    color: str | None = None,
    dash: str = "solid",
    width: float = 1.5,
    opacity: float | None = None,
    mode: str = "lines"  # <-- new parameter, can be "lines", "lines+markers", etc.
) -> go.Scatter:
    """
    Create a standardised line trace for Plotly.

    This function generates a reusable Plotly Scatter trace configured as a
    line chart. It supports custom colours, line styles, widths, opacity,
    and scatter modes (e.g., lines with markers). The trace is designed for
    consistent styling across visualisations.

    Parameters
    ----------
    x : pd.Series
        X-axis values (e.g., dates).
    y : pd.Series
        Y-axis values (metric values).
    name : str
        Name of the trace (used in legend).
    color : str, optional
        Line colour (default: None).
    dash : str, optional
        Line style: "solid", "dot", "dash", etc. (default: "solid").
    width : float, optional
        Line width (default: 1.5).
    opacity : float, optional
        Opacity of the trace (default: None).
    mode : str, optional
        Plotly scatter mode ("lines", "lines+markers", etc.) (default: "lines").

    Returns
    -------
    go.Scatter
        Plotly Scatter trace configured as a line (with optional markers).

    Notes
    -----
    - Supports flexible styling for consistent visualisation.
    - The default colour was previously "#1f77b4".
    - Opacity is applied only if explicitly provided.
    """
    trace = go.Scatter(
        x=x,
        y=y,
        mode=mode,
        name=name,
        line=dict(color=color, dash=dash, width=width)
    )
    if opacity is not None:
        trace.opacity = opacity
    return trace


def build_heatmap_figure(
    pivot: pd.DataFrame,
    title: str = "Heatmap",
    xaxis_title: str = "X Axis",
    yaxis_title: str = "Y Axis",
    hovertemplate: str | None = None,
    texttemplate: str = "%{text}",
    colorscale: str = "YlGnBu",
    autorange: str | None = "reversed",
    width: int = 1080,
    height: int = 500
) -> go.Figure:
    """
    Build a Plotly heatmap figure from a pivot table.

    This function creates a heatmap visualisation from a pivot table of
    numeric values. It supports custom titles, axis labels, hover text,
    annotation styles, colour scales, and layout dimensions. The y-axis can
    optionally be reversed for better readability.

    Parameters
    ----------
    pivot : pd.DataFrame
        Pivot table containing numeric values.
        - Index : rows.
        - Columns : columns.
    title : str, optional
        Title of the plot (default: "Heatmap").
    xaxis_title : str, optional
        Label for the x-axis (default: "X Axis").
    yaxis_title : str, optional
        Label for the y-axis (default: "Y Axis").
    hovertemplate : str, optional
        Custom hover text. If None, defaults to "X: %{x}<br>Y: %{y}<br>Value: %{z:.2f}".
    texttemplate : str, optional
        Annotation style inside cells (default: "%{text}").
    colorscale : str, optional
        Colour scheme for the heatmap (default: "YlGnBu").
    autorange : str, optional
        Y-axis autorange setting (default: "reversed").
    width : int, optional
        Width of the figure in pixels (default: 1080).
    height : int, optional
        Height of the figure in pixels (default: 500).

    Returns
    -------
    go.Figure
        Plotly heatmap figure object.

    Notes
    -----
    - Numeric values are converted to floats before plotting.
    - Hover tooltips display x, y, and z values with two decimal precision.
    - Cell annotations show rounded values unless NaN.
    - Layout styling is applied via `apply_common_layout`.
    - Interactive features include hover tooltips, zoom, pan, and export options.
    """

    # Ensure numeric values
    z_values = pivot.astype(float).values

    # Default hovertemplate
    if hovertemplate is None:
        hovertemplate = "X: %{x}<br>Y: %{y}<br>Value: %{z:.2f}<extra></extra>"

    # Create heatmap trace
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=colorscale,
            text=np.where(np.isnan(z_values), "", np.round(z_values, 2)),
            texttemplate=texttemplate,
            hovertemplate=hovertemplate,
            zmin=np.nanmin(z_values),
            zmax=np.nanmax(z_values),
            showscale=True
        )
    )

    # Apply common layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        showlegend=False
    )

    # Override axis specifics for heatmap
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange=autorange)
    )

    return fig


def get_df_summary(df: pd.Series) -> pd.Series:
    """
    Return summary statistics of product diversity per order.

    This function computes descriptive statistics for a Series representing
    product diversity (e.g., number of unique products per invoice). It
    leverages pandas’ `describe()` method to provide a quick overview of the
    distribution.

    Parameters
    ----------
    df : pd.Series
        Series containing product diversity values per invoice.

    Returns
    -------
    pd.Series
        Summary statistics including:
        - count : number of observations.
        - mean : average product diversity.
        - std : standard deviation.
        - min : minimum value.
        - 25% : first quartile.
        - 50% : median.
        - 75% : third quartile.
        - max : maximum value.

    Notes
    -----
    - Useful for understanding variability in product diversity across orders.
    - Provides a concise statistical profile without additional computation.
    """
    return df.describe()


def price_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of unit price statistics for each product (StockCode).

    This function groups the input DataFrame by product code and computes
    descriptive statistics for unit prices. It provides measures of central
    tendency, dispersion, and variability, along with counts of unique and
    total price observations. The relative standard deviation highlights
    products with unstable or highly variable pricing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifiers.
        - 'UnitPrice' : price per unit.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per StockCode and the following columns:
        - 'StockCode' : product identifier.
        - 'mean_unitprice_by_stockcode' : mean unit price.
        - 'median_unitprice_by_stockcode' : median unit price.
        - 'mode_unitprice_by_stockcode' : mode unit price (first mode if multiple).
        - 'std_unitprice_by_stockcode' : standard deviation of unit price.
        - 'min_unitprice_by_stockcode' : minimum unit price.
        - 'max_unitprice_by_stockcode' : maximum unit price.
        - 'unique_unitprice_by_stockcode' : number of unique unit prices observed.
        - 'count_unitprice_by_stockcode' : total number of unit price observations.
        - 'relative_std_unitprice_by_stockcode' : relative standard deviation (std ÷ mean).

    Notes
    -----
    - The function groups the DataFrame by 'StockCode' and computes descriptive statistics for 'UnitPrice'.
    - The relative standard deviation (RSD) is useful for identifying products with high price variability.
    - Can be extended to identify products with the highest RSD if needed.
    - Works seamlessly in both Jupyter notebooks and Streamlit apps.
    """
    summary = df.groupby("StockCode")["UnitPrice"].agg(
        mean_unitprice_by_stockcode="mean",
        median_unitprice_by_stockcode="median",
        mode_unitprice_by_stockcode=lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None,
        std_unitprice_by_stockcode="std",
        min_unitprice_by_stockcode="min",
        max_unitprice_by_stockcode="max",
        unique_unitprice_by_stockcode="nunique"
    ).reset_index()

    summary["count_unitprice_by_stockcode"] = df.groupby("StockCode")["UnitPrice"].count().values
    summary["relative_std_unitprice_by_stockcode"] = (
        summary["std_unitprice_by_stockcode"] / summary["mean_unitprice_by_stockcode"]
    )

    return summary

def top_products_by_rsd(summary: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Select the top products ranked by relative standard deviation of unit price.

    This function identifies products with the highest variability in unit
    prices by ranking them according to their relative standard deviation.
    It returns the top N products, sorted in descending order, making it
    useful for spotting items with inconsistent pricing.

    Parameters
    ----------
    summary : pd.DataFrame
        Input DataFrame containing at least:
        - 'relative_std_unitprice_by_stockcode' : relative standard deviation of unit prices per product.
        Typically, this DataFrame is a product-level summary with computed statistics.
    n : int, optional
        Number of products to return (default: 10).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the top N products with the highest values of
        'relative_std_unitprice_by_stockcode', sorted in descending order.

    Notes
    -----
    - The function uses `nlargest()` to efficiently select the top N rows.
    - 'relative_std_unitprice_by_stockcode' represents the relative standard deviation of unit prices for each product (stock code).
    - Useful for identifying products with the most price variability.
    - Works seamlessly in both Jupyter notebooks and Streamlit apps.
    """
    return summary.nlargest(n, "relative_std_unitprice_by_stockcode")

def price_distribution_by_code(summary: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    Compute the distribution of unit prices for a given stock code.

    This function filters the input DataFrame by a specified stock code and
    computes the frequency of each unique unit price. The resulting summary
    is useful for analysing pricing patterns and preparing visualisations
    such as histograms or bar chart    def graph_blended_results(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=range(len(self.y_test)), y=self.y_test, label="Baseline", ax=ax)
        sns.lineplot(x=range(len(self.pred_ridge)), y=self.pred_ridge, color="red", label="Ridge", alpha=0.7, ax=ax)
        sns.lineplot(x=range(len(self.pred_rf)), y=self.pred_rf, color="black", label="Random Forest", alpha=0.7, ax=ax)
        sns.lineplot(x=range(len(self.blend)), y=self.blend, color="green", label="Best Blend", alpha=0.7, ax=ax)
        return figs.

    Parameters
    ----------
    summary : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product codes.
        - 'UnitPrice' : price per unit.
    stock_code : str
        The stock code to filter by and compute the price distribution.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - 'UnitPrice' : unique unit price values for the given stock code.
        - 'count' : frequency of each unit price.

    Notes
    -----
    - The DataFrame is filtered to include only rows matching the specified stock code.
    - Frequency of each unit price is computed using `value_counts()`.
    - The resulting DataFrame is useful for further visualisation (e.g., histograms or bar charts).
    - Works seamlessly with both Jupyter notebooks and Streamlit apps.
    """
    dist = summary.loc[summary["StockCode"] == stock_code, "UnitPrice"].value_counts().reset_index()
    dist.columns = ["UnitPrice", "count"]
    return dist


def plot_price_distribution(df: pd.DataFrame, stock_code: str) -> go.Figure:
    """
    Plot a histogram of unit prices for a specified stock code using Plotly.

    This function filters the input DataFrame by a given stock code and
    visualises the distribution of unit prices as a histogram. It provides
    insight into the pricing spread for a specific product.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product codes.
        - 'UnitPrice' : price per unit.
    stock_code : str
        The stock code to filter by and visualise the unit price distribution.

    Returns
    -------
    go.Figure
        Plotly histogram showing the distribution of unit prices for the given stock code.

    Notes
    -----
    - The DataFrame is filtered to include only rows matching the specified stock code.
    - A histogram of 'UnitPrice' values is plotted with 30 bins by default.
    - Axis labels are set to "Unit Price" (x-axis) and "Count" (y-axis).
    - The plot title is automatically generated as "Price distribution for <stock_code>".
    - Interactive features include hover tooltips, zoom, pan, and export options.
    """

    filtered = df[df["StockCode"] == stock_code]

    fig = px.histogram(
        filtered,
        x="UnitPrice",
        nbins=30,
        title=f"Price distribution for {stock_code}"
    )

    # Apply common layout helper
    fig = apply_common_layout(
        fig,
        title=f"Price distribution for {stock_code}",
        xaxis_title="Unit Price",
        yaxis_title="Count",
        showlegend=False  # histogram doesn’t need a legend
    )

    # Add histogram-specific styling
    fig.update_layout(bargap=0.05)

    return fig


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a country-level summary of revenue and quantity metrics.

    This function aggregates transactional data by country and computes
    descriptive statistics, invoice counts, per-invoice averages, and
    relative shares of revenue and quantity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the following columns:
        - 'Country': country names.
        - 'Quantity': number of items purchased.
        - 'Revenue': total revenue values.
        - 'InvoiceNo': invoice identifiers.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per country and the following columns:
        - 'mean_quantity_by_country': mean quantity of items per country.
        - 'median_quantity_by_country': median quantity of items per country.
        - 'total_quantity_by_country': total quantity of items per country.
        - 'mean_revenue_by_country': mean revenue per country.
        - 'median_revenue_by_country': median revenue per country.
        - 'total_revenue_by_country': total revenue per country.
        - 'unique_invoice_by_country': number of unique invoices per country.
        - 'mean_revenue_per_invoice_by_country': average revenue per invoice.
        - 'mean_quantity_per_invoice_by_country': average quantity per invoice.
        - 'mean_revenue_per_quantity_by_country': average revenue per unit quantity.
        - 'share_of_revenue_by_country': proportion of total revenue contributed by the country.
        - 'share_of_quantity_by_country': proportion of total quantity contributed by the country.

    Notes
    -----
    - Division by zero is handled by replacing denominators with NaN where necessary.
    - Shares are computed relative to the overall dataset totals.
    - Useful for comparative analysis of country-level performance.
    """
    summary = df.groupby("Country").agg(
        mean_quantity_by_country = ("Quantity", "mean"),
        median_quantity_by_country = ("Quantity", "median"),
        total_quantity_by_country = ("Quantity", "sum"),

        mean_revenue_by_country = ("Revenue", "mean"),
        median_revenue_by_country = ("Revenue", "median"),
        total_revenue_by_country = ("Revenue", "sum"),

        unique_invoice_by_country = ("InvoiceNo", "nunique")
    ).reset_index()

    summary["mean_revenue_per_invoice_by_country"] = (
            summary["total_revenue_by_country"] / summary["unique_invoice_by_country"].replace(0, np.nan)
    )
    summary["mean_quantity_per_invoice_by_country"] = (
            summary["total_quantity_by_country"] / summary["unique_invoice_by_country"].replace(0, np.nan)
    )
    summary["mean_revenue_per_quantity_by_country"] = (
            summary["total_revenue_by_country"] / summary["total_quantity_by_country"].replace(0, np.nan)
    )
    summary["share_of_revenue_by_country"] = (
            summary["total_revenue_by_country"] / summary["total_revenue_by_country"].sum()
    )
    summary["share_of_quantity_by_country"] = (
            summary["total_quantity_by_country"] / summary["total_quantity_by_country"].sum()
    )

    return summary


def top_countries_by(summary: pd.DataFrame, column: str, n: int | None = None, ascending: bool = False) -> pd.DataFrame:
    """
    Return the top or bottom countries by a chosen summary column.

    This function sorts countries by a specified column and returns either
    the top or bottom N countries. If N is not provided, the full sorted
    table is returned. A ValueError is raised if the chosen column does not
    exist in the input DataFrame.

    Parameters
    ----------
    summary : pd.DataFrame
        Input DataFrame containing country-level summary information.
    column : str
        Column name to rank countries by (e.g., 'total_revenue_by_country').
    n : int, optional
        Number of countries to return. If None, include all countries (default: None).
    ascending : bool, optional
        Sort order. False = descending (largest values first, default).
        True = ascending (smallest values first).

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with either the top/bottom N countries or all countries.
    """

    if column not in summary.columns:
        raise ValueError(
            f"Column '{column}' not found in summary. "
            f"Available columns are: {list(summary.columns)}"
        )

    sorted_summary = summary.sort_values(column, ascending=ascending)

    if n is None:
        return sorted_summary
    else:
        return sorted_summary.head(n)


def plot_top_countries(summary: pd.DataFrame,
                       column: str,
                       n: int | None = None,
                       ascending: bool = False,
                       exclude: list[str] | None = None,
                       log_scale: bool = False,
                       title: str | None = None) -> go.Figure:
    """
    Plot a bar chart of the top or bottom countries ranked by a chosen summary column.

    Parameters
    ----------
    summary : pd.DataFrame
        Country-level summary DataFrame containing at least:
        - 'Country': country names.
        - <column>: numeric values to rank countries by.
    column : str
        Column name to use for ranking (e.g., 'total_revenue_by_country').
    n : int or None, optional
        Number of countries to display. If None, include all countries.
    ascending : bool, optional
        Sort order for ranking. If False (default), show top n countries.
        If True, show bottom n countries.
    exclude : list of str or None, optional
        List of countries to exclude from the plot (e.g., ['United Kingdom']).
    log_scale : bool, optional
        If True, apply logarithmic scaling to the y-axis to handle skewed distributions.
    title : str or None, optional
        Plot title. If None, automatically generated from the column name.

    Returns
    -------
    go.Figure
        A Plotly figure object showing a bar chart of the selected countries
        ranked by the specified summary column.

    Notes
    -----
    - Excluded countries are removed before ranking and plotting.
    - Sorting is based on the specified column, with control over ascending/descending order.
    - Supports both linear and logarithmic y-axis scaling.
    - Interactive features include hover tooltips, zoom, pan, and export options.
    """

    if exclude:
        summary = summary[~summary["Country"].isin(exclude)]

    data = top_countries_by(summary, column, n=n, ascending=ascending)

    fig = px.bar(
        data,
        x="Country",
        y=column,
        title=title or f"{column.replace('_', ' ').title()} by Country",
        text=column,
        color="Country",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title=column.replace("_by_country", "").replace("_", " ").title(),
        xaxis_tickangle=-45,
        width=1080,
        height=500,
        showlegend=False,
    )

    if log_scale:
        fig.update_layout(yaxis_type="log")

    return fig


def plot_country_treemap(summary: pd.DataFrame,
                         column: str = "total_revenue_by_country",
                         exclude: list[str] | None = None,
                         title: str | None = None) -> go.Figure:
    """
    Plot a treemap visualization of country-level contributions for a chosen summary column.

    Parameters
    ----------
    summary : pd.DataFrame
        Country-level summary DataFrame containing at least:
        - 'Country': country names.
        - <column>: numeric values to visualise.
    column : str, optional
        Column to display in the treemap (default: 'total_revenue_by_country').
    exclude : list of str or None, optional
        List of countries to exclude from the treemap (e.g., ['United Kingdom']).
    title : str or None, optional
        Title of the plot. If None, automatically generated from the column name.

    Returns
    -------
    go.Figure
        A Plotly figure object showing a treemap of countries sized by the specified column.

    Notes
    -----
    - Excluded countries are removed before visualisation.
    - Each rectangle represents a country, sized proportionally to its value in the chosen column.
    - Interactive features include hover tooltips, zoom, pan, and export options.
    - Useful for visualising relative contributions of countries to overall totals.
    """

    if exclude:
        summary = summary[~summary["Country"].isin(exclude)]

    fig = px.treemap(
        summary,
        path=["Country"],
        values=column,
        color="Country",  # categorical coloring for distinct countries
        title=title or f"Share of {column.replace('_by_country','').replace('_',' ').title()} by Country"
    )

    fig.update_layout(
        width=1080,
        height=500
    )

    return fig


def customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of customer-level metrics.

    This function aggregates customer-level data to compute descriptive
    statistics and derived metrics for both quantity and revenue. It
    summarises mean, median, and total values, as well as transaction counts
    and per-transaction averages, providing a comprehensive view of customer
    behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'CustomerID' : unique customer identifiers.
        - 'Quantity' : number of items purchased.
        - 'Revenue' : monetary value of purchases.
        - 'InvoiceNo' : transaction identifiers.

    Returns
    -------
    pd.DataFrame
        Customer-level summary with aggregated and derived metrics, including:
        - 'mean_quantity_by_customer' : average quantity purchased.
        - 'median_quantity_by_customer' : median quantity purchased.
        - 'total_quantity_by_customer' : total quantity purchased.
        - 'mean_revenue_by_customer' : average revenue generated.
        - 'median_revenue_by_customer' : median revenue generated.
        - 'total_revenue_by_customer' : total revenue generated.
        - 'unique_transactions_by_customer' : number of unique transactions.
        - 'mean_revenue_per_transaction_by_customer' : average revenue per transaction.
        - 'mean_quantity_per_transaction_by_customer' : average quantity per transaction.
        - 'mean_revenue_per_item_by_customer' : average revenue per item purchased.
    """
    summary = (
        df[df["CustomerID"] != 0]
        .groupby("CustomerID")
        .agg(
            mean_quantity_by_customer=("Quantity", "mean"),
            median_quantity_by_customer=("Quantity", "median"),
            total_quantity_by_customer=("Quantity", "sum"),

            mean_revenue_by_customer=("Revenue", "mean"),
            median_revenue_by_customer=("Revenue", "median"),
            total_revenue_by_customer=("Revenue", "sum"),

            unique_transactions_by_customer=("InvoiceNo", "nunique")
        )
        .reset_index()
    )

    # Derived metrics
    summary["mean_revenue_per_transaction_by_customer"] = (
        summary["total_revenue_by_customer"] / summary["unique_transactions_by_customer"].replace(0, np.nan)
    )
    summary["mean_quantity_per_transaction_by_customer"] = (
        summary["total_quantity_by_customer"] / summary["unique_transactions_by_customer"].replace(0, np.nan)
    )
    summary["mean_revenue_per_item_by_customer"] = (
        summary["total_revenue_by_customer"] / summary["total_quantity_by_customer"].replace(0, np.nan)
    )

    return summary


def top_customers_by(summary: pd.DataFrame,
                     column: str,
                     n: int | None = None,
                     ascending: bool = False) -> pd.DataFrame:
    """
    Return the top or bottom customers by a chosen summary column.

    This function sorts customers by a specified column and returns either
    the top or bottom N customers. If N is not provided, the full sorted
    table is returned. A ValueError is raised if the chosen column does not
    exist in the input DataFrame.

    Parameters
    ----------
    summary : pd.DataFrame
        Input DataFrame containing customer summary information.
    column : str
        Column name to rank customers by (e.g., 'total_revenue_by_customer').
    n : int, optional
        Number of customers to return. If None, include all customers (default: None).
    ascending : bool, optional
        Sort order. False = descending (largest values first, default).
        True = ascending (smallest values first).

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with either the top/bottom N customers or all customers.
    """
    if column not in summary.columns:
        raise ValueError(
            f"Column '{column}' not found in summary. "
            f"Available columns are: {list(summary.columns)}"
        )

    sorted_summary = summary.sort_values(column, ascending=ascending)

    if n is None:
        return sorted_summary
    else:
        return sorted_summary.head(n)


def filter_customers_by_transactions(summary: pd.DataFrame,
                                     transactions: int = 1,
                                     sort_by: str = "total_revenue",
                                     ascending: bool = False,
                                     n: int | None = None) -> pd.DataFrame:
    """
    Filter customers by transaction count and sort results.

    This function returns customers with exactly the specified number of
    transactions, sorted by a chosen column. It also supports limiting the
    output to the top N customers. If the chosen sort column does not exist
    in the input DataFrame, a ValueError is raised.

    Parameters
    ----------
    summary : pd.DataFrame
        Input DataFrame containing at least:
        - 'unique_transactions_by_customer' : number of transactions per customer.
        - 'total_revenue' : total revenue per customer (used for sorting by default).
    transactions : int, optional
        Number of transactions to filter customers by (default: 1).
    sort_by : str, optional
        Column to sort the results by (default: 'total_revenue').
    ascending : bool, optional
        Sort order. False = descending (default), True = ascending.
    n : int, optional
        Number of customers to return. If None, include all (default: None).

    Returns
    -------
    pd.DataFrame
        Subset of customers filtered by transaction count,
        sorted by the chosen column, optionally limited to N rows.
    """
    if sort_by not in summary.columns:
        raise ValueError(
            f"Column '{sort_by}' not found in summary. "
            f"Available columns are: {list(summary.columns)}"
        )

    filtered = summary[summary["unique_transactions_by_customer"] == transactions]
    sorted_filtered = filtered.sort_values(by=sort_by, ascending=ascending)

    if n is None:
        return sorted_filtered
    else:
        return sorted_filtered.head(n)


def customer_distribution_by_country(df: pd.DataFrame,
                                     n: int | None = None,
                                     exclude: list[str] | None = None) -> pd.DataFrame:
    """
    Generate a summary table of customer distribution by country.

    This function aggregates customer counts by country, excluding invalid
    customer IDs, and provides options to exclude specific countries or
    limit the output to the top N countries. The summary is sorted by
    customer counts in descending order by default.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'CustomerID' : unique customer identifiers.
        - 'Country' : country values associated with each customer.
    n : int, optional
        Number of countries to return. If None, include all (default: None).
    exclude : list[str], optional
        List of countries to exclude from the summary (e.g., ['United Kingdom']).

    Returns
    -------
    pd.DataFrame
        Country-level summary with:
        - 'Country' : country name.
        - 'total_customers_by_country' : total unique customers per country.
        Sorted by customer counts in descending order.
    """
    summary = (
        df[df["CustomerID"] != 0]
        .groupby("Country")
        .agg(total_customers_by_country=("CustomerID", "nunique"))
        .reset_index()
    )

    # Apply exclusions if provided
    if exclude:
        summary = summary[~summary["Country"].isin(exclude)]

    # Sort by total customers
    summary = summary.sort_values(by="total_customers_by_country", ascending=False)

    # Limit to top N if requested
    if n is not None:
        summary = summary.head(n)

    return summary


def plot_customer_distribution_by_country(
        df: pd.DataFrame,
        n: int | None = None,
        ascending: bool = False,
        exclude: list[str] | None = None,
        title: str | None = None,
        orientation: str = "v",
        log_scale: bool = False
) -> go.Figure:
    """
    Plot the distribution of customers by country as a bar chart.

    This function summarises customer counts by country and visualises them
    using Plotly. It supports both vertical and horizontal orientations,
    optional exclusion of countries, limiting to the top N countries, and
    logarithmic scaling. Layout styling is standardised via `apply_common_layout`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing customer and country information.
    n : int, optional
        Number of top countries to display; if None, include all (default: None).
    ascending : bool, optional
        Sort order for customer counts; False for descending (default: False).
    exclude : list[str], optional
        List of countries to exclude from the plot (default: None).
    title : str, optional
        Custom plot title; if None, a default title is used (default: None).
    orientation : str, optional
        Bar orientation, either "v" (vertical) or "h" (horizontal) (default: "v").
    log_scale : bool, optional
        Whether to use a logarithmic scale for the value axis (default: False).

    Returns
    -------
    go.Figure
        Plotly bar chart showing customer distribution by country.
    """

    summary = customer_distribution_by_country(df, n=n, exclude=exclude)
    summary = summary.sort_values(by="total_customers_by_country", ascending=ascending)

    if orientation == "h":
        fig = px.bar(
            summary,
            x="total_customers_by_country",
            y="Country",
            orientation="h",
            text="total_customers_by_country",
            color="Country"
        )
        fig.update_traces(textposition="outside")

        # Use helper for layout
        fig = apply_common_layout(
            fig,
            title=title or "Customer Distribution by Country",
            xaxis_title="Total Customers",
            yaxis_title="Country",
            width=1080,
            height=((n * 30) + 200) if n else 1080,
            showlegend=False,
            grids="x"
        )
        fig.update_xaxes(type="log" if log_scale else "linear")

    else:
        fig = px.bar(
            summary,
            x="Country",
            y="total_customers_by_country",
            orientation="v",
            text="total_customers_by_country",
            color="Country"
        )
        fig.update_traces(textposition="outside")

        # Use helper for layout
        fig = apply_common_layout(
            fig,
            title=title or "Customer Distribution by Country",
            xaxis_title="Country",
            yaxis_title="Total Customers",
            width=1080,
            height=500,
            showlegend=False,
            grids="y"
        )
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(type="log" if log_scale else "linear")

    return fig


def daywise_summary_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a day-wise summary for both Revenue and Quantity
    with rolling, cumulative, grouped averages, and gap flags.

    This function prepares a comprehensive daily summary of sales metrics,
    including totals, rolling and cumulative averages, grouped averages
    (weekly, monthly, quarterly, yearly), and flags for missing data periods.
    It provides a structured dataset for further analysis and visualisation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'date' : date values.
        - 'Revenue' : daily revenue totals.
        - 'Quantity' : daily units sold totals.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date with the following columns for both metrics:
        - 'Revenue', 'Quantity' : daily totals.
        - 'cumulative_mean_<metric>_by_day' : cumulative average up to each day.
        - 'rolling_7d_mean_<metric>_by_day' : 7-day rolling average.
        - 'rolling_30d_mean_<metric>_by_day' : 30-day rolling average.
        - 'mean_<metric>_by_week' : weekly average.
        - 'mean_<metric>_by_month' : monthly average.
        - 'mean_<metric>_by_quarter' : quarterly average.
        - 'mean_<metric>_by_year' : yearly average.
        - 'is_missing_<metric>_by_day' : flag for missing values.
        - 'gap_start_<metric>_by_day' : flag for start of missing gap.
        - 'gap_end_<metric>_by_day' : flag for end of missing gap.
    """

    # Aggregate daily totals for both metrics
    daywise = (
        df.groupby(by="date", sort=False)
        .agg({"Revenue": "sum", "Quantity": "sum"})
        .reset_index()
    )

    # Ensure full date range continuity
    full_range = pd.date_range(start=daywise["date"].min(), end=daywise["date"].max())
    daywise_full = (
        daywise.set_index("date")
        .reindex(full_range)
        .rename_axis("date")
        .reset_index()
    )

    for col in ["Revenue", "Quantity"]:
        col_lower = col.lower()

        # Rolling and cumulative averages
        daywise_full[f"cumulative_mean_{col_lower}_by_day"] = daywise_full[col].expanding().mean()
        daywise_full[f"rolling_7d_mean_{col_lower}_by_day"] = daywise_full[col].rolling(7, min_periods=1).mean()
        daywise_full[f"rolling_30d_mean_{col_lower}_by_day"] = daywise_full[col].rolling(30, min_periods=1).mean()

        # Grouped averages
        daywise_full[f"mean_{col_lower}_by_week"] = daywise_full.groupby(pd.Grouper(key="date", freq="W"))[col].transform("mean")
        daywise_full[f"mean_{col_lower}_by_month"] = daywise_full.groupby(pd.Grouper(key="date", freq="ME"))[col].transform("mean")
        daywise_full[f"mean_{col_lower}_by_quarter"] = daywise_full.groupby(pd.Grouper(key="date", freq="QE"))[col].transform("mean")
        daywise_full[f"mean_{col_lower}_by_year"] = daywise_full.groupby(pd.Grouper(key="date", freq="YE"))[col].transform("mean")

        # Missing data flags
        daywise_full[f"is_missing_{col_lower}_by_day"] = daywise_full[col].isna()
        daywise_full[f"gap_start_{col_lower}_by_day"] = (
            daywise_full[f"is_missing_{col_lower}_by_day"]
            & ~daywise_full[f"is_missing_{col_lower}_by_day"].shift(1, fill_value=False)
        )
        daywise_full[f"gap_end_{col_lower}_by_day"] = (
            daywise_full[f"is_missing_{col_lower}_by_day"]
            & ~daywise_full[f"is_missing_{col_lower}_by_day"].shift(-1, fill_value=False)
        )

    return daywise_full


# Revenue plots
def plot_daywise_metric_with_gaps(
    daywise_data: pd.DataFrame,
    column: str = "Revenue",
    title: str | None = None,
    show_gaps: bool = True
) -> go.Figure:
    """
    Plot day-wise metric (Revenue or Quantity) with missing periods shaded using Plotly.

    This function visualises daily values of a chosen metric (Revenue or Quantity)
    over time, highlighting missing periods with shaded regions. It provides a clear
    view of trends while marking gaps in the data for better interpretation.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        Input DataFrame containing at least:
        - 'date' : date values.
        - '<column>' : daily values for the chosen metric.
        - 'gap_start_<column.lower()>_by_day' : boolean flag indicating the start of a missing gap.
        - 'gap_end_<column.lower()>_by_day' : boolean flag indicating the end of a missing gap.
    column : str, optional
        Column to plot. Must be either "Revenue" or "Quantity" (default: "Revenue").
    title : str, optional
        Title of the plot (default: "Daily <column> with Missing Periods Shaded").
    show_gaps : bool, optional
        Whether to highlight missing periods with shaded regions (default: True).

    Returns
    -------
    go.Figure
        Plotly line chart showing daily values with shaded regions or markers indicating missing periods.
    """

    col_lower = column.lower()
    if title is None:
        title = f"Daily {column} with Missing Periods Shaded"

    # Extract gap intervals
    gap_starts = daywise_data.loc[daywise_data[f"gap_start_{col_lower}_by_day"], "date"]
    gap_ends = daywise_data.loc[daywise_data[f"gap_end_{col_lower}_by_day"], "date"]

    # Base figure
    fig = go.Figure()

    # Add main line trace using helper
    fig.add_trace(
        make_line_trace(
            x=daywise_data["date"],
            y=daywise_data[column],
            name=f"Daily {column}",
            mode="lines+markers"
        )
    )

    if show_gaps:
        # Shade missing intervals
        for start, end in zip(gap_starts, gap_ends):
            if start == end:
                # one-day gap → vertical line
                fig.add_shape(
                    type="line",
                    x0=start,
                    x1=start,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="red", width=2),
                    opacity=0.2
                )
            else:
                # multi-day gap → shaded span
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text="",
                    showlegend=False
                )

        # Add a single dummy trace for legend entry
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="rgba(255, 0, 0, 0.2)"),
            name="Missing Data"
        )

    # Apply common layout using helper
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Date",
        yaxis_title=f"Daily {column}"
    )

    return fig


def plot_average_daily_revenue(
    daywise_data: pd.DataFrame,
    title: str = "Average Daily Revenue"
) -> go.Figure:
    """
    Plot average daily revenue and its moving averages using Plotly.

    This function visualises daily revenue trends alongside cumulative,
    rolling (7-day and 30-day), and overall averages. It provides a clear
    view of short-term fluctuations and longer-term revenue patterns.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        Input DataFrame containing at least:
        - 'date' : date values.
        - 'Revenue' : daily revenue totals.
        - 'cumulative_mean_revenue_by_day' : cumulative average of daily revenue.
        - 'rolling_7d_mean_revenue_by_day' : 7-day rolling average of daily revenue.
        - 'rolling_30d_mean_revenue_by_day' : 30-day rolling average of daily revenue.
    title : str, optional
        Title of the plot (default: "Average Daily Revenue").

    Returns
    -------
    go.Figure
        Plotly line chart showing daily revenue and averages.
    """

    fig = go.Figure()

    # Total Daily Revenue
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["Revenue"],
        name="Total Daily Revenue"
    ))

    # Cumulative Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["cumulative_mean_revenue_by_day"],
        name="Cumulative Average of Daily Revenue",
        opacity=0.7
    ))

    # 7-day Rolling Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["rolling_7d_mean_revenue_by_day"],
        name="7-day Rolling Average of Daily Revenue"
    ))

    # 30-day Rolling Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["rolling_30d_mean_revenue_by_day"],
        name="30-day Rolling Average of Daily Revenue"
    ))

    # Overall Average (horizontal dotted line)
    avg_value = daywise_data["Revenue"].mean()
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=[avg_value] * len(daywise_data),
        name="Average of All Daily Revenue",
        dash="dot"
    ))

    # Apply common layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Date",
        yaxis_title="Revenue"
    )

    return fig


def plot_timely_avg_revenue(
    daywise_data: pd.DataFrame,
    title: str = "Timely Averages of Daily Revenue"
) -> go.Figure:
    """
    Plot weekly, monthly, quarterly, and yearly averages of daily revenue using Plotly.

    This function visualises revenue trends over different time horizons,
    including weekly, monthly, quarterly, and yearly averages. It also
    displays the overall average as a reference line, allowing comparison
    between short-term and long-term revenue patterns.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        Input DataFrame containing at least:
        - 'date' : date values.
        - 'Revenue' : daily revenue totals.
        - 'mean_revenue_by_week' : weekly average of daily revenue.
        - 'mean_revenue_by_month' : monthly average of daily revenue.
        - 'mean_revenue_by_quarter' : quarterly average of daily revenue.
        - 'mean_revenue_by_year' : yearly average of daily revenue.
    title : str, optional
        Title of the plot (default: "Timely Averages of Daily Revenue").

    Returns
    -------
    go.Figure
        Plotly line chart showing weekly, monthly, quarterly, yearly,
        and overall averages of daily revenue.
    """

    fig = go.Figure()

    # Weekly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_revenue_by_week"],
        name="Weekly Average of Daily Revenue"
    ))

    # Monthly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_revenue_by_month"],
        name="Monthly Average of Daily Revenue",
        dash="dash"
    ))

    # Quarterly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_revenue_by_quarter"],
        name="Quarterly Average of Daily Revenue",
        dash="dashdot"
    ))

    # Yearly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_revenue_by_year"],
        name="Yearly Average of Daily Revenue"
    ))

    # Overall Average (horizontal dotted line)
    avg_value = daywise_data["Revenue"].mean()
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=[avg_value] * len(daywise_data),
        name="Average of All Daily Revenue",
        dash="dot"
    ))

    # Apply common layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Date",
        yaxis_title="Total Revenue"
    )

    return fig


def plot_revenue_histogram(
    daywise_data: pd.DataFrame,
    title: str = "Histogram of Daily Total Revenue",
    bin_size: int = 5000
) -> go.Figure:
    """
    Plot a histogram of daily total revenue with a KDE curve using Plotly.

    This function visualises the distribution of daily revenue values
    through histogram bars and overlays a kernel density estimate (KDE)
    curve to highlight the underlying distribution pattern. It provides
    insight into the spread and central tendency of daily revenue.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        DataFrame containing daily revenue values. Must include a 'Revenue' column.
    title : str, optional
        Title of the plot (default: "Histogram of Daily Total Revenue").
    bin_size : int, optional
        Width of histogram bins in revenue units (default: 5000).

    Returns
    -------
    go.Figure
        Plotly figure showing the histogram of daily revenue with a KDE curve.
    """

    # Create histogram + KDE curve
    fig = ff.create_distplot(
        [daywise_data["Revenue"].dropna()],
        group_labels=["Revenue"],
        bin_size=bin_size,
        show_hist=True,
        show_curve=True
    )

    # Style histogram bars with borders
    fig.update_traces(
        selector=dict(type="histogram"),
        marker=dict(
            color="steelblue",
            opacity=0.7,
            line=dict(color="black", width=1)  # border around bars
        )
    )

    # Apply layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title=f"Daily Revenue",
        yaxis_title="Density",
        showlegend=False
    )

    return fig


# Quantity plots
def plot_average_daily_quantity(
    daywise_data: pd.DataFrame,
    title: str = "Average Daily Quantity"
) -> go.Figure:
    """
    Plot average daily quantity and its moving averages using Plotly.

    This function visualises daily quantity trends alongside cumulative,
    rolling (7-day and 30-day), and overall averages. It provides a clear
    view of short-term fluctuations and longer-term quantity patterns.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        Input DataFrame containing at least:
        - 'date' : date values.
        - 'Quantity' : daily units sold totals.
        - 'cumulative_mean_quantity_by_day' : cumulative average of daily quantity.
        - 'rolling_7d_mean_quantity_by_day' : 7-day rolling average of daily quantity.
        - 'rolling_30d_mean_quantity_by_day' : 30-day rolling average of daily quantity.
    title : str, optional
        Title of the plot (default: "Average Daily Quantity").

    Returns
    -------
    go.Figure
        Plotly line chart showing daily quantity and averages.
    """

    fig = go.Figure()

    # Total Daily Quantity
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["Quantity"],
        name="Total Daily Quantity"
    ))

    # Cumulative Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["cumulative_mean_quantity_by_day"],
        name="Cumulative Average of Daily Quantity",
        opacity=0.7
    ))

    # 7-day Rolling Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["rolling_7d_mean_quantity_by_day"],
        name="7-day Rolling Average of Daily Quantity"
    ))

    # 30-day Rolling Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["rolling_30d_mean_quantity_by_day"],
        name="30-day Rolling Average of Daily Quantity"
    ))

    # Overall Average (horizontal dotted line)
    avg_value = daywise_data["Quantity"].mean()
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=[avg_value] * len(daywise_data),
        name="Average of All Daily Quantity",
        dash="dot"
    ))

    # Apply common layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Date",
        yaxis_title="Quantity",
        legend_position=(0.2, 0.96)
    )

    return fig


def plot_timely_avg_quantity(
    daywise_data: pd.DataFrame,
    title: str = "Timely Averages of Daily Quantity"
) -> go.Figure:
    """
    Plot weekly, monthly, quarterly, and yearly averages of daily quantity using Plotly.

    This function visualises quantity trends over different time horizons,
    including weekly, monthly, quarterly, and yearly averages. It also
    displays the overall average as a reference line, enabling comparison
    between short-term and long-term quantity patterns.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        Input DataFrame containing at least:
        - 'date' : date values.
        - 'Quantity' : daily units sold totals.
        - 'mean_quantity_by_week' : weekly average of daily quantity.
        - 'mean_quantity_by_month' : monthly average of daily quantity.
        - 'mean_quantity_by_quarter' : quarterly average of daily quantity.
        - 'mean_quantity_by_year' : yearly average of daily quantity.
    title : str, optional
        Title of the plot (default: "Timely Averages of Daily Quantity").

    Returns
    -------
    go.Figure
        Plotly line chart showing weekly, monthly, quarterly, yearly,
        and overall averages of daily quantity.
    """

    fig = go.Figure()

    # Weekly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_quantity_by_week"],
        name="Weekly Average of Daily Quantity"
    ))

    # Monthly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_quantity_by_month"],
        name="Monthly Average of Daily Quantity",
        dash="dash"
    ))

    # Quarterly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_quantity_by_quarter"],
        name="Quarterly Average of Daily Quantity",
        dash="dashdot"
    ))

    # Yearly Average
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["mean_quantity_by_year"],
        name="Yearly Average of Daily Quantity"
    ))

    # Overall Average (horizontal dotted line)
    avg_value = daywise_data["Quantity"].mean()
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=[avg_value] * len(daywise_data),
        name="Average of All Daily Quantity",
        dash="dot"
    ))

    # Apply common layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Date",
        yaxis_title="Total Quantity"
    )

    return fig


def plot_quantity_histogram(
    daywise_data: pd.DataFrame,
    title: str = "Histogram of Daily Total Quantity",
    bin_size: int = 2500
) -> go.Figure:
    """
    Plot a histogram of daily total quantity with a KDE curve using Plotly.

    This function visualises the distribution of daily quantity values
    through histogram bars and overlays a kernel density estimate (KDE)
    curve to highlight the underlying distribution pattern. It provides
    insight into the spread and central tendency of daily quantity.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        DataFrame containing daily quantity values. Must include a 'Quantity' column.
    title : str, optional
        Title of the plot (default: "Histogram of Daily Total Quantity").
    bin_size : int, optional
        Width of histogram bins in quantity units (default: 2500).

    Returns
    -------
    go.Figure
        Plotly figure showing the histogram of daily quantity with a KDE curve.
    """

    # Create histogram + KDE curve
    fig = ff.create_distplot(
        [daywise_data["Quantity"].dropna()],
        group_labels=["Quantity"],
        bin_size=bin_size,
        show_hist=True,
        show_curve=True
    )

    # Style histogram bars with borders
    fig.update_traces(
        selector=dict(type="histogram"),
        marker=dict(
            color="steelblue",
            opacity=0.7,
            line=dict(color="black", width=1)  # border around bars
        )
    )

    # Apply layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title=f"Daily Quantity",
        yaxis_title="Density",
        showlegend=False
    )

    return fig


# Basic Heatmaps
def plot_revenue_heatmap_by_weekday_hour(
    df: pd.DataFrame,
    title: str = "Heatmap of Average Revenue by Day of Week and Hour of Day"
) -> go.Figure:
    """
    Plot a heatmap of average revenue by day of week and hour of day using Plotly.

    This function visualises patterns in revenue across different weekdays
    and hours of the day. It highlights when revenue is typically higher or
    lower, providing insight into temporal trends and customer behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Revenue' : daily revenue totals.
        - 'day_of_week' : categorical weekday values.
        - 'hour' : hour of the day.
    title : str, optional
        Title of the plot (default: "Heatmap of Average Revenue by Day of Week and Hour of Day").

    Returns
    -------
    go.Figure
        Plotly heatmap showing average revenue by weekday and hour.
    """

    pivot = df.pivot_table(
        values="Revenue",
        index="day_of_week",
        columns="hour",
        aggfunc="mean",
        observed=True
    )

    fig = build_heatmap_figure(
        pivot,
        title=title,
        xaxis_title="Hour of the Day",
        yaxis_title="Day of Week",
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>Revenue: %{z:.2f}<extra></extra>"
    )

    return fig


def plot_quantity_heatmap_by_weekday_hour(
    df: pd.DataFrame,
    title: str = "Heatmap of Average Quantity by Day of Week and Hour of Day"
) -> go.Figure:
    """
    Plot a heatmap of average quantity by day of week and hour of day using Plotly.

    This function visualises patterns in quantity across different weekdays
    and hours of the day. It highlights when sales volume is typically higher
    or lower, providing insight into temporal trends and customer behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Quantity' : daily units sold totals.
        - 'day_of_week' : categorical weekday values.
        - 'hour' : hour of the day.
    title : str, optional
        Title of the plot (default: "Heatmap of Average Quantity by Day of Week and Hour of Day").

    Returns
    -------
    go.Figure
        Plotly heatmap showing average quantity by weekday and hour.
    """

    pivot = df.pivot_table(
        values="Quantity",
        index="day_of_week",
        columns="hour",
        aggfunc="mean",
        observed=True
    )

    fig = build_heatmap_figure(
        pivot,
        title=title,
        xaxis_title="Hour of the Day",
        yaxis_title="Day of Week",
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>Quantity: %{z:.2f}<extra></extra>"
    )

    return fig


def plot_revenue_heatmap_by_weekday_month(
    df: pd.DataFrame,
    title: str = "Heatmap of Average Revenue by Day of Week and Month"
) -> go.Figure:
    """
    Plot a heatmap of average revenue by day of week and month using Plotly.

    This function visualises patterns in revenue across different weekdays
    and months. It highlights when revenue is typically higher or lower,
    providing insight into seasonal and weekly trends in customer behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Revenue' : daily revenue totals.
        - 'day_of_week' : categorical weekday values.
        - 'month' : categorical month values.
    title : str, optional
        Title of the plot (default: "Heatmap of Average Revenue by Day of Week and Month").

    Returns
    -------
    go.Figure
        Plotly heatmap showing average revenue by weekday and month.
    """

    pivot = df.pivot_table(
        values="Revenue",
        index="day_of_week",
        columns="month",
        aggfunc="mean",
        observed=True
    )

    fig = build_heatmap_figure(
        pivot,
        title=title,
        xaxis_title="Month",
        yaxis_title="Day of Week",
        hovertemplate="Month: %{x}<br>Day: %{y}<br>Revenue: %{z:.2f}<extra></extra>"
    )

    return fig


def plot_quantity_heatmap_by_weekday_month(
    df: pd.DataFrame,
    title: str = "Heatmap of Average Quantity by Day of Week and Month"
) -> go.Figure:
    """
    Plot a heatmap of average quantity by day of week and month using Plotly.

    This function visualises patterns in sales quantity across different weekdays
    and months. It highlights when sales volume is typically higher or lower,
    providing insight into seasonal and weekly trends in customer behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Quantity' : daily units sold totals.
        - 'day_of_week' : categorical weekday values.
        - 'month' : categorical month values.
    title : str, optional
        Title of the plot (default: "Heatmap of Average Quantity by Day of Week and Month").

    Returns
    -------
    go.Figure
        Plotly heatmap showing average quantity by weekday and month.
    """

    pivot = df.pivot_table(
        values="Quantity",
        index="day_of_week",
        columns="month",
        aggfunc="mean",
        observed=True
    )

    fig = build_heatmap_figure(
        pivot,
        title=title,
        xaxis_title="Month",
        yaxis_title="Day of Week",
        hovertemplate="Month: %{x}<br>Day: %{y}<br>Quantity: %{z:.2f}<extra></extra>"
    )

    return fig


# Revenue vs Quantity
def plot_revenue_vs_quantity_scatter(
    daywise_data: pd.DataFrame,
    title: str = "Scatterplot of Total Daily Revenue vs Daily Units Sold"
) -> go.Figure:
    """
    Plot a scatterplot comparing daily revenue and daily quantity sold using Plotly.

    This function visualises the relationship between total daily revenue
    and the number of units sold. Each point represents one day’s data,
    allowing patterns, correlations, or anomalies between sales volume and
    revenue to be identified.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        Input DataFrame containing at least:
        - 'Revenue' : daily revenue totals.
        - 'Quantity' : daily units sold.
    title : str, optional
        Title of the plot (default: "Scatterplot of Total Daily Revenue vs Daily Units Sold").

    Returns
    -------
    go.Figure
        Plotly scatterplot showing the relationship between daily revenue and units sold.
    """

    fig = go.Figure()

    # Scatter points (markers only)
    fig.add_trace(
        go.Scatter(
            x=daywise_data["Revenue"],
            y=daywise_data["Quantity"],
            mode="markers",
            marker=dict(
                size=8,
                color="#1f77b4",
                opacity=1,
                line=dict(width=0.5, color="white")
            ),
            name="Daily Data Points"
        )
    )

    # Apply common layout helper
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Daily Revenue",
        yaxis_title="Units Sold",
        height=600,
        showlegend=False
    )

    return fig


def plot_revenue_and_quantity_over_time(
    daywise_data: pd.DataFrame,
    title: str = "Daily Revenue and Units Sold Over Time"
) -> go.Figure:
    """
    Plot daily revenue and quantity on the same scale as a dual line chart using Plotly.

    This function visualises daily revenue and units sold together over time,
    allowing direct comparison of sales volume and revenue trends. It displays
    revenue as a solid line and quantity as a dotted line, making it easy to
    observe correlations and divergences between the two measures.

    Parameters
    ----------
    daywise_data : pd.DataFrame
        Input DataFrame containing at least:
        - 'date' : date values.
        - 'Revenue' : daily revenue totals.
        - 'Quantity' : daily units sold.
    title : str, optional
        Title of the plot (default: "Daily Revenue and Units Sold Over Time").

    Returns
    -------
    go.Figure
        Plotly dual line chart showing daily revenue and units sold over time.
    """

    fig = go.Figure()

    # Revenue line (solid blue)
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["Revenue"],
        name="Revenue",
        color="#1f77b4"
    ))

    # Quantity line (green dotted)
    fig.add_trace(make_line_trace(
        x=daywise_data["date"],
        y=daywise_data["Quantity"],
        name="Units Sold",
        color="green",
        dash="dot"
    ))

    # Apply common layout
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Date",
        yaxis_title="Values"
    )

    return fig


def plot_metric_heatmap_by_weekday_month_premium(
    df: pd.DataFrame,
    title: str = "Heatmap of Average Revenue per Unit Sold"
) -> go.Figure:
    """
    Plot a heatmap of average revenue per unit sold by day of week and month using Plotly.

    This function calculates the premium metric (revenue divided by quantity)
    across weekdays and months, and visualises it as a heatmap. It highlights
    patterns in how much revenue is generated per unit sold, providing insight
    into customer behaviour and seasonal variations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Revenue' : daily revenue totals.
        - 'Quantity' : daily units sold totals.
        - 'day_of_week' : categorical weekday values.
        - 'month' : categorical month values.
    title : str, optional
        Title of the plot (default: "Heatmap of Average Revenue per Unit Sold").

    Returns
    -------
    go.Figure
        Plotly heatmap showing average revenue per unit sold by weekday and month.
    """

    # Calculate premium pivot
    pivot = df.pivot_table(
        values=["Revenue", "Quantity"],
        index="day_of_week",
        columns="month",
        aggfunc="mean",
        observed=True
    )
    premium_pivot = pivot["Revenue"] / pivot["Quantity"]

    # Build heatmap using the core builder
    fig = build_heatmap_figure(
        premium_pivot,
        title=title,
        xaxis_title="Month",
        yaxis_title="Day of Week",
        hovertemplate="Month: %{x}<br>Day: %{y}<br>Revenue per Unit: %{z:.2f}<extra></extra>"
    )

    return fig


def plot_metric_heatmap_by_weekday_hour_premium(
    df: pd.DataFrame,
    title: str = "Heatmap of Average Revenue per Unit Sold"
) -> go.Figure:
    """
    Plot a heatmap of average revenue per unit sold by day of week and hour of day using Plotly.

    This function calculates the premium metric (revenue divided by quantity)
    across weekdays and hours of the day, and visualises it as a heatmap. It
    highlights patterns in how much revenue is generated per unit sold at
    different times, providing insight into customer behaviour and temporal
    variations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Revenue' : daily revenue totals.
        - 'Quantity' : daily units sold totals.
        - 'day_of_week' : categorical weekday values.
        - 'hour' : hour of the day.
    title : str, optional
        Title of the plot (default: "Heatmap of Average Revenue per Unit Sold").

    Returns
    -------
    go.Figure
        Plotly heatmap showing average revenue per unit sold by weekday and hour.
    """

    # Calculate premium pivot
    pivot = df.pivot_table(
        values=["Revenue", "Quantity"],
        index="day_of_week",
        columns="hour",
        aggfunc="mean",
        observed=True
    )
    premium_pivot = pivot["Revenue"] / pivot["Quantity"]

    # Build heatmap using the core builder
    fig = build_heatmap_figure(
        premium_pivot,
        title=title,
        xaxis_title="Hour of the Day",
        yaxis_title="Day of Week",
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>Revenue per Unit: %{z:.2f}<extra></extra>"
    )

    return fig


def plot_revenue_boxplot_by_weekday(
    df: pd.DataFrame,
    title: str = "Boxplot of Revenue across Days of the Week",
    orientation: str = "v"
) -> go.Figure:
    """
    Plot a boxplot of daily revenue across days of the week using Plotly.

    This function visualises the distribution of daily revenue for each
    weekday using boxplots. Outliers flagged in the input data are excluded,
    ensuring a clearer view of the underlying distribution. Each box shows
    the median, quartiles, whiskers, and includes mean and standard deviation
    lines, allowing comparison of revenue patterns across weekdays.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Revenue' : numeric daily revenue values.
        - 'day_of_week' : categorical weekday labels.
        - 'is_outlier' : boolean flag for outliers.
    title : str, optional
        Title of the plot (default: "Boxplot of Revenue across Days of the Week").
    orientation : {"v", "h"}, optional
        Orientation of the boxplots:
        - "v" (default): vertical boxes, with days on the x-axis and revenue on the y-axis.
        - "h": horizontal boxes, with revenue on the x-axis and days on the y-axis.

    Returns
    -------
    go.Figure
        Plotly boxplot showing the distribution of daily revenue across weekdays.
    """

    # Filter out outliers
    filtered = df[df["is_outlier"] == False]

    fig = go.Figure()

    # Build boxplot traces
    for day in filtered["day_of_week"].unique():
        day_data = filtered.loc[filtered["day_of_week"] == day, "Revenue"]

        if orientation == "h":
            fig.add_trace(
                go.Box(
                    x=day_data,
                    y=[day] * len(day_data),
                    name=day,
                    boxmean="sd",
                    marker=dict(opacity=0.7),
                    orientation="h"
                )
            )
            fig.update_layout(
                yaxis=dict(categoryorder="array", categoryarray=df["day_of_week"].cat.categories, autorange="reversed")
            )
        else:  # default vertical
            fig.add_trace(
                go.Box(
                    x=[day] * len(day_data),
                    y=day_data,
                    name=day,
                    boxmean="sd",
                    marker=dict(opacity=0.7),
                    orientation="v"
                )
            )
            fig.update_layout(
                xaxis=dict(categoryorder="array", categoryarray=df["day_of_week"].cat.categories)
            )

    # Apply common layout helper
    fig = apply_common_layout(
        fig,
        title=title,
        xaxis_title="Revenue" if orientation == "h" else "Day of Week",
        yaxis_title="Day of Week" if orientation == "h" else "Revenue",
        showlegend=False
    )

    return fig


def plot_avg_metrics_comparison(
        df: pd.DataFrame,
        group_col: str = "day_of_week",
        title: str | None = None
) -> go.Figure:
    """
    Compare average Revenue and Quantity by a chosen categorical column (e.g., weekday, month).

    The function computes the average Revenue and Quantity grouped by the specified
    categorical column, then renders two bar traces on dual y-axes with the same scale
    so their relative sizes are easy to compare. Revenue is shown with a blue palette
    on the left axis, and Quantity with a green palette on the right axis. Values are
    annotated on the bars with two-decimal precision.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - group_col : categorical or string column used for grouping (e.g., 'day_of_week', 'month').
        - 'Revenue' : numeric revenue values.
        - 'Quantity' : numeric units-sold values.
    group_col : str, optional
        Column name to group by (default: 'day_of_week').
    title : str, optional
        Plot title. If not provided, a default title is generated based on the grouping column.

    Returns
    -------
    go.Figure
        Plotly Figure with dual y-axes showing overlapping bar traces for Revenue and Quantity.
    """

    # Aggregate: average metrics by weekday (renamed variable as requested)
    avg_metric = (
        df.groupby(group_col, observed=False)
        .agg({"Revenue": "mean", "Quantity": "mean"})
        .reset_index()
    ).dropna(subset=["Revenue", "Quantity"])

    # Prepare x and y values
    x = avg_metric[group_col]
    revenue_y = avg_metric["Revenue"].round(2)
    quantity_y = avg_metric["Quantity"].round(2)

    # Determine a common scale range
    max_val = max(revenue_y.max(), quantity_y.max())
    y_range = [0, max_val * 1.1]  # Add a little headroom

    # Build figure with overlapping bars (overlay) so bars visually compare on same scale
    fig = go.Figure()

    # Revenue
    fig.add_trace(
        go.Bar(
            x=x,
            y=revenue_y,
            name="Revenue",
            marker=dict(color="#33759f"),
            text=[f"{v:.2f}" for v in revenue_y],
            textposition="outside",
            textfont=dict(color="#33759f"),
            hovertemplate="%{x}<br>Revenue: %{y:.2f}<extra></extra>",
            yaxis="y"
        )
    )

    # Quantity
    fig.add_trace(
        go.Bar(
            x=x,
            y=quantity_y,
            name="Units Sold",
            marker=dict(color="#9ce29c"),
            text=[f"{v:.2f}" for v in quantity_y],
            textposition="outside",
            textfont=dict(color="#9ce29c"),
            hovertemplate="%{x}<br>Units Sold: %{y:.2f}<extra></extra>",
            yaxis="y2"
        )
    )

    fig.update_layout(
        title=title or f"Average of Revenue and Units Sold by {group_col.replace('_', ' ').title()}",
        xaxis=dict(title=group_col.replace("_", " ").title()),
        yaxis=dict(
            title=dict(text="Revenue", font=dict(color="#07467b")),
            tickfont=dict(color="#33759f"),
            range=y_range
        ),
        yaxis2=dict(
            title=dict(text="Quantity", font=dict(color="green")),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
            range=y_range
        ),
        template="plotly_white",
        width=1080,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02)
    )

    try:
        cats = avg_metric[group_col].cat.categories
        fig.update_layout(xaxis=dict(categoryorder="array", categoryarray=list(cats)))
    except Exception:
        pass

    return fig


def plot_orders_by_time(
    df: pd.DataFrame,
    group_col: str = "hour",
    title: str | None = None
) -> go.Figure:
    """
    Plot a bar chart of unique orders grouped by a time-related column using Plotly.

    This function counts the number of unique orders (based on `InvoiceNo`)
    grouped by a specified time-related column (such as hour, weekday, or month)
    and visualises them as a bar chart. Each bar is uniquely coloured, making it
    easy to compare order volumes across different time intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'InvoiceNo' : unique order identifier.
        - group_col : column to group by (e.g., 'hour', 'day_of_week', 'month').
    group_col : str, optional
        Column name to group by (default: 'hour').
    title : str, optional
        Title of the plot. If not provided, a default title is generated based
        on the grouping column.

    Returns
    -------
    go.Figure
        Plotly bar chart showing the number of unique orders grouped by the
        specified time-related column.
    """

    # Aggregate: count unique orders by chosen grouping column
    orders_grouped = (
        df.groupby(group_col, observed=False)
        .agg({"InvoiceNo": "nunique"})
        .reset_index()
    ).dropna(subset=["InvoiceNo"])

    x = orders_grouped[group_col]
    y = orders_grouped["InvoiceNo"]

    # Build figure
    fig = go.Figure()

    # Each bar gets a unique color via marker.color sequence
    # Add one trace per category so Plotly uses default colors
    for xi, yi in zip(x, y):
        fig.add_trace(
            go.Bar(
                x=[xi],
                y=[yi],
                name=str(xi),
                text=[f"{yi:d}"],
                textposition="outside",
                hovertemplate=f"{xi}<br>Orders: {yi}<extra></extra>"
            )
        )

    # Layout
    fig.update_layout(
        title=title or f"Orders placed by {group_col.replace('_',' ').title()}",
        xaxis=dict(title=group_col.replace("_"," ").title()),
        yaxis=dict(title="Total Orders Placed"),
        template="plotly_white",
        width=1080,
        height=500,
        showlegend=False
    )

    # Respect categorical order if defined
    try:
        cats = orders_grouped[group_col].cat.categories
        fig.update_layout(xaxis=dict(categoryorder="array", categoryarray=list(cats)))
    except Exception:
        pass

    return fig


def plot_orders_heatmap(
    df: pd.DataFrame,
    row_col: str = "day_of_week",
    col_col: str = "hour",
    title: str = "Heatmap of Orders"
) -> go.Figure:
    """
    Plot a heatmap of unique orders grouped by two categorical columns using Plotly.

    This function counts the number of unique orders (based on `InvoiceNo`)
    grouped by two specified categorical columns (such as weekday vs hour or
    weekday vs month) and visualises them as a heatmap. It highlights patterns
    in order volumes across different time dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'InvoiceNo' : unique order identifier.
        - row_col : column to use for rows (default: 'day_of_week').
        - col_col : column to use for columns (default: 'hour').
    row_col : str, optional
        Column name to use for rows (default: 'day_of_week').
    col_col : str, optional
        Column name to use for columns (default: 'hour').
    title : str, optional
        Title of the plot (default: "Heatmap of Orders").

    Returns
    -------
    go.Figure
        Plotly heatmap showing the number of unique orders grouped by the
        specified row and column categories.
    """

    # Build pivot table
    pivot = df.pivot_table(
        values="InvoiceNo",
        index=row_col,
        columns=col_col,
        aggfunc="nunique",
        observed=True
    )

    # Call the helper function
    fig = build_heatmap_figure(
        pivot=pivot,
        title=title,
        xaxis_title=col_col.replace("_", " ").title(),
        yaxis_title=row_col.replace("_", " ").title(),
        texttemplate="%{text}",
        hovertemplate=f"{col_col.title()}: %{{x}}<br>{row_col.title()}: %{{y}}<br>Orders: %{{z:.0f}}<extra></extra>"
    )

    return fig


def plot_rpo_heatmap(
    df: pd.DataFrame,
    row_col: str = "day_of_week",
    col_col: str = "hour",
    exclude_single_orders: bool = False,
    title: str | None = None
) -> go.Figure:
    """
    Plot a heatmap of average revenue per order (RPO) grouped by two categorical columns using Plotly.

    This function calculates the average revenue per order by dividing total
    revenue by the number of unique orders, grouped across two specified
    categorical columns (such as weekday vs hour or weekday vs month). It
    visualises the results as a heatmap, highlighting patterns in order value
    across different time dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Revenue' : numeric revenue values.
        - 'InvoiceNo' : unique order identifier.
        - row_col : column to use for rows (default: 'day_of_week').
        - col_col : column to use for columns (default: 'hour').
    row_col : str, optional
        Column name to use for rows (default: 'day_of_week').
    col_col : str, optional
        Column name to use for columns (default: 'hour').
    exclude_single_orders : bool, optional
        If True, cells where only one order was placed are excluded (default: False).
    title : str, optional
        Title of the plot. If not provided, a default title is generated.

    Returns
    -------
    go.Figure
        Plotly heatmap showing average revenue per order grouped by the specified
        row and column categories.
    """

    # Pivot tables
    pivot_revenue = df.pivot_table(
        values="Revenue", index=row_col, columns=col_col,
        aggfunc="sum", observed=True
    )
    pivot_orders = df.pivot_table(
        values="InvoiceNo", index=row_col, columns=col_col,
        aggfunc="nunique", observed=True
    )

    # Compute Revenue per Order
    if exclude_single_orders:
        pivot_orders = pivot_orders.where(pivot_orders > 1)
    rpo_pivot = pivot_revenue / pivot_orders

    # Build heatmap
    fig = build_heatmap_figure(
        pivot=rpo_pivot,
        title=title or f"Heatmap of Average Revenue per Order by {row_col.replace('_',' ').title()} and {col_col.replace('_',' ').title()}",
        xaxis_title=col_col.replace("_", " ").title(),
        yaxis_title=row_col.replace("_", " ").title(),
        texttemplate="%{text}",
        hovertemplate=f"{col_col.title()}: %{{x}}<br>{row_col.title()}: %{{y}}<br>Avg Revenue/Order: %{{z:.2f}}<extra></extra>"
    )

    return fig


def plot_qpo_heatmap(
    df: pd.DataFrame,
    row_col: str = "day_of_week",
    col_col: str = "month",
    exclude_single_orders: bool = False,
    title: str | None = None
) -> go.Figure:
    """
    Plot a heatmap of average units sold per order (QPO) grouped by two categorical columns using Plotly.

    This function calculates the average units sold per order by dividing total
    quantity by the number of unique orders, grouped across two specified
    categorical columns (such as weekday vs month or weekday vs hour). It
    visualises the results as a heatmap, highlighting patterns in order size
    across different time dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Quantity' : numeric units sold.
        - 'InvoiceNo' : unique order identifier.
        - row_col : column to use for rows (default: 'day_of_week').
        - col_col : column to use for columns (default: 'month').
    row_col : str, optional
        Column name to use for rows (default: 'day_of_week').
    col_col : str, optional
        Column name to use for columns (default: 'month').
    exclude_single_orders : bool, optional
        If True, cells where only one order was placed are excluded (default: False).
    title : str, optional
        Title of the plot. If not provided, a default title is generated.

    Returns
    -------
    go.Figure
        Plotly heatmap showing average units sold per order grouped by the specified
        row and column categories.
    """

    # Pivot tables
    pivot_quantity = df.pivot_table(
        values="Quantity", index=row_col, columns=col_col,
        aggfunc="sum", observed=True
    )
    pivot_orders = df.pivot_table(
        values="InvoiceNo", index=row_col, columns=col_col,
        aggfunc="nunique", observed=True
    )

    # Compute QPO (Quantity per Order)
    if exclude_single_orders:
        pivot_orders = pivot_orders.where(pivot_orders > 1)
    qpo_pivot = pivot_quantity / pivot_orders

    # Build heatmap using your helper
    fig = build_heatmap_figure(
        pivot=qpo_pivot,
        title=title or f"Heatmap of Average Units Sold per Order by {row_col.replace('_',' ').title()} and {col_col.replace('_',' ').title()}",
        xaxis_title=col_col.replace("_", " ").title(),
        yaxis_title=row_col.replace("_", " ").title(),
        texttemplate="%{text}",
        hovertemplate=f"{col_col.title()}: %{{x}}<br>{row_col.title()}: %{{y}}<br>Avg Units/Order: %{{z:.2f}}<extra></extra>",
    )

    return fig


# Product-Level Insights
def get_top_bottom_products(
    df: pd.DataFrame,
    sort_col: str,
    n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Aggregate product-level metrics and return the top or bottom n entries based on a specified column.

    This function groups data by product (`StockCode`) and calculates total
    revenue, total quantity sold, and the number of unique orders. It also
    extracts the most frequent product description for each code. The results
    are sorted by a chosen metric and the top or bottom n entries are returned.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifier.
        - 'Revenue' : numeric revenue values.
        - 'Quantity' : numeric units sold.
        - 'InvoiceNo' : unique order identifier.
        - 'Description' : product description.
    sort_col : str
        Column name to sort by ('Revenue', 'Quantity', or 'Orders').
    n : int, optional
        Number of entries to return (default: 10).
    ascending : bool, optional
        Sort order. If True, returns the bottom n entries.
        If False, returns the top n entries (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame of aggregated product metrics sorted by the chosen column,
        including the most frequent description for each product.
    """

    # Aggregate product-level metrics
    product_df = (
        df.groupby("StockCode")
        .agg({
            "Revenue": "sum",
            "Quantity": "sum",
            "InvoiceNo": "nunique",
            "Description": lambda x: x.value_counts().idxmax() if len(x) > 0 else None
        })
        .reset_index()
    )
    product_df.rename(columns={"InvoiceNo": "Orders"}, inplace=True)

    # Sort and select top/bottom n
    result = product_df.sort_values(by=sort_col, ascending=ascending).head(n)
    result["Description"] = result["Description"].str.title()

    return result


#Time-Based Popularity
def build_monthly_product_sales(df: pd.DataFrame, description_mode: dict) -> pd.DataFrame:
    """
    Build a DataFrame of monthly product sales with descriptions and additional statistics.

    This function aggregates product-level revenue by month and attaches the
    most frequent description for each product using a precomputed mapping.
    It also calculates the relative standard deviation (RSD) of revenue per
    product and computes each product’s share of revenue within its month.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifier.
        - 'Revenue' : numeric revenue values.
        - 'month' : month identifier (numeric or categorical).
        - 'Description' : product description.
    description_mode : dict
        Dictionary mapping each StockCode to its most frequent Description
        (title-cased). Typically generated once during data cleaning.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'StockCode' : product identifier.
        - 'month' : month of aggregation.
        - 'Revenue' : total revenue per product per month.
        - 'Description' : most frequent description for each product (title-cased).
        - 'RSD' : relative standard deviation of revenue per product.
        - 'Revenue_Share' : share of product revenue within its month.

    Notes
    -----
    - The description mapping is passed in as a parameter to avoid recomputation.
    - RSD is computed as standard deviation ÷ mean revenue per product.
    - Revenue share is calculated relative to the total revenue of each month.
    - Useful for analysing product performance, variability, and contribution
      to monthly revenue.
    """

    # Build description_mode mapping #DEPRECATED
    # description_mode_series = (
    #     df.groupby("StockCode")["Description"].apply(
    #         lambda x: x.mode().iloc[0] if x.shape[0] > 2 and not x.mode().empty else None
    #     )
    # )
    # description_mode = description_mode_series.dropna().to_dict()

    # Aggregate monthly product sales
    monthly_product_sales = (
        df.groupby(["StockCode", "month"], observed=False)["Revenue"].sum().reset_index()
    )

    # Map most frequent description (title-cased)
    monthly_product_sales["Description"] = (
        monthly_product_sales["StockCode"].map(description_mode).str.title()
    )

    # Compute mean, std, and RSD for each StockCode
    stats = monthly_product_sales.groupby("StockCode")["Revenue"].agg(["mean", "std"])
    stats["RSD"] = np.round(stats["std"] / stats["mean"], 4)

    # Merge RSD back into monthly_product_sales
    monthly_product_sales = monthly_product_sales.merge(stats["RSD"], on="StockCode", how="left")

    # Compute revenue share within each month
    monthly_product_sales["Revenue_Share"] = (
        monthly_product_sales.groupby("month", observed=False)["Revenue"]
        .transform(lambda x: x / x.sum())
    )

    return monthly_product_sales


def top_products_by_month(monthly_product_sales: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """
    Return the top n products with the highest revenue for each month.

    This function selects the top products by revenue from a precomputed
    monthly product sales DataFrame. It groups data by month, identifies the
    highest-revenue products, and returns the top n entries for each month.

    Parameters
    ----------
    monthly_product_sales : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifier.
        - 'month' : month identifier.
        - 'Revenue' : numeric revenue values.
        - 'Description' : product description (optional).
    n : int, optional
        Number of top products to return per month (default: 1).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the top n products by revenue for each month,
        sorted by month.
    """

    top_products = (
        monthly_product_sales.groupby("month", observed=False)
        .apply(lambda g: g.nlargest(n, "Revenue"))
        .reset_index(drop=True)
        .sort_values(by="month")
    )

    return top_products


def top_volatile_products_by_rsd(
    monthly_product_sales: pd.DataFrame,
    description_mode: dict,
    n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Return the top or bottom products ranked by volatility (RSD).

    This function aggregates product-level revenue and mean relative standard
    deviation (RSD) of revenue, then ranks products by volatility. It attaches
    the most frequent description for each product using a precomputed mapping
    to avoid recomputation.

    Parameters
    ----------
    monthly_product_sales : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifier.
        - 'Revenue' : numeric revenue values.
        - 'RSD' : relative standard deviation of revenue per StockCode.
        - 'Description' : product description.
    description_mode : dict
        Dictionary mapping each StockCode to its most frequent Description
        (title-cased). Typically generated once during data cleaning.
    n : int, optional
        Number of products to return (default: 10).
    ascending : bool, optional
        Sort order for RSD. False (default) returns highest volatility.
        True returns lowest volatility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the top or bottom N products ranked by volatility (RSD),
        with columns:
        - 'StockCode' : product identifier.
        - 'Revenue' : aggregated revenue across months.
        - 'RSD' : mean relative standard deviation of revenue per product.
        - 'Description' : most frequent description for each product (title-cased).

    Notes
    -----
    - Description mapping is passed in as a parameter to avoid recomputation.
    - Products are sorted by RSD, with Revenue used as a tiebreaker.
    - Useful for identifying products with the most or least revenue variability.
    """

    # Aggregate revenue and mean RSD per StockCode
    volatility_rsd = (
        monthly_product_sales.groupby("StockCode", observed=False)
        .agg({"Revenue": "sum", "RSD": "mean"})
        .reset_index()
    )

    # Attach Description using precomputed dictionary
    if description_mode is not None:
        volatility_rsd["Description"] = volatility_rsd["StockCode"].map(description_mode).str.title()

    # Sort by RSD (and Revenue as tiebreaker), then select top/bottom n
    result = (
        volatility_rsd.sort_values(by=["RSD", "Revenue"], ascending=[ascending, False])
        .head(n)
        .reset_index(drop=True)
    )

    return result


def plot_monthly_revenue_heatmap(
    monthly_product_sales: pd.DataFrame,
    n: int = 20
) -> go.Figure:
    """
    Plot a heatmap of monthly revenue for the top products.

    This function selects the top N products ranked by total revenue and
    visualises their monthly revenue patterns as a heatmap. It provides a
    quick way to identify seasonality, revenue concentration, and product
    performance trends across months.

    Parameters
    ----------
    monthly_product_sales : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifier.
        - 'month' : month identifier (numeric or categorical).
        - 'Revenue' : numeric revenue values.
    n : int, optional
        Number of top products to include (default: 20).

    Returns
    -------
    go.Figure
        Plotly heatmap figure showing monthly revenue distribution for the
        top N products.

    Notes
    -----
    - Products are ranked by total revenue across all months before selection.
    - The DataFrame is pivoted to create a matrix of StockCode vs month.
    - Missing values are filled with 0 to ensure a complete heatmap.
    - The figure is built using `build_heatmap_figure` for consistent styling.
    - Height of the figure scales with the number of products to maintain readability.
    """

    # Select top n products by total revenue
    top_products = (
        monthly_product_sales.groupby("StockCode")["Revenue"].sum().nlargest(n).index
    )

    # Filter for those products
    filtered = monthly_product_sales[monthly_product_sales["StockCode"].isin(top_products)]

    # Pivot to create matrix of StockCode vs month
    seasonality = (
        filtered.pivot(index="StockCode", columns="month", values="Revenue").fillna(0)
    )

    # Build Plotly heatmap figure using helper
    fig = build_heatmap_figure(
        pivot=seasonality,
        title=f"Monthly Revenue Heatmap by Product (Top {n} products with most revenue)",
        xaxis_title="Month",
        yaxis_title="StockCode",
        height=((n*30) + 200)
    )

    return fig


def plot_top_product_monthly_revenue_trend(
    monthly_product_sales: pd.DataFrame,
    description_mode: dict
) -> go.Figure:
    """
    Plot monthly revenue trend for the product with the highest total revenue.

    This function identifies the product with the highest total revenue across
    all months and visualises its monthly revenue trend using a combined bar
    and line chart. The product’s most frequent description is attached via a
    precomputed mapping to provide context in the plot title.

    Parameters
    ----------
    monthly_product_sales : pd.DataFrame
        Input DataFrame containing at least:
        - 'StockCode' : product identifier.
        - 'month' : month identifier (numeric or categorical).
        - 'Revenue' : numeric revenue values.
    description_mode : dict
        Dictionary mapping each StockCode to its most frequent Description
        (title-cased). Typically generated once during data cleaning.

    Returns
    -------
    go.Figure
        Plotly figure combining:
        - Bar chart of monthly revenue values.
        - Line chart showing revenue trend over months.

    Notes
    -----
    - The product with the highest total revenue is selected automatically.
    - The description mapping is used to display the product name in the plot title.
    - Bar trace shows monthly revenue values with labels, while the line trace
      highlights the overall trend.
    - Layout styling is applied via `apply_common_layout` for consistency.
    """

    # Identify product with highest total revenue directly from monthly_product_sales
    most_revenue_code = (
        monthly_product_sales.groupby("StockCode")["Revenue"].sum().idxmax()
    )

    # Filter for that product
    product_trend = monthly_product_sales.loc[
        monthly_product_sales["StockCode"] == most_revenue_code
    ]

    # Bar trace for monthly revenue
    bar_trace = go.Bar(
        x=product_trend["month"],
        y=product_trend["Revenue"],
        name="Revenue",
        text=product_trend["Revenue"].round(2),
        textposition="outside",
        marker=dict(color="steelblue")
    )

    # Line trace using helper
    line_trace = make_line_trace(
        x=product_trend["month"],
        y=product_trend["Revenue"],
        name="Revenue Trend",
        color="lightgreen",
        mode="lines+markers",
        width=2
    )

    # Build figure
    fig = go.Figure(data=[bar_trace, line_trace])

    # Apply common layout (using defaults for width/height)
    fig = apply_common_layout(
        fig,
        title=f"Monthly Revenue Trend for Product '{description_mode[most_revenue_code]}'",
        xaxis_title="Month",
        yaxis_title="Total Revenue"
    )

    return fig


# Product Diversity
def build_diversity_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Calculate product diversity per invoice.

    This function computes the number of unique products purchased in each
    invoice by grouping transactions on `InvoiceNo` and counting distinct
    `StockCode` values. It provides a simple measure of product diversity
    per order.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'InvoiceNo' : invoice identifiers.
        - 'StockCode' : product codes.

    Returns
    -------
    pd.Series
        Series indexed by InvoiceNo with values representing the number of
        unique products in each order.

    Notes
    -----
    - Uses `nunique()` to count distinct product codes per invoice.
    - Useful for analysing customer purchasing behaviour and order diversity.
    """

    return pd.Series(df.groupby("InvoiceNo")["StockCode"].nunique())


def get_top_diverse_invoices(
    distribution: pd.Series,
    n: int = 10,
    ascending: bool = False
) -> pd.Series:
    """
    Return the top invoices ranked by product diversity.

    This function sorts a Series of product diversity values (number of unique
    products per invoice) and returns the top or bottom N invoices depending
    on the sort order. It provides a quick way to identify the most or least
    diverse orders.

    Parameters
    ----------
    distribution : pd.Series
        Series containing product diversity per invoice.
    n : int, optional
        Number of invoices to return (default: 10).
    ascending : bool, optional
        Sort order for product diversity. False (default) returns the most
        diverse orders. True returns the least diverse orders.

    Returns
    -------
    pd.Series
        Series containing the top or bottom N invoices sorted by product
        diversity.

    Notes
    -----
    - Diversity is measured as the number of unique products per invoice.
    - Sorting is performed on the values, with head(n) selecting the top N.
    - Useful for identifying customers with broad or narrow purchasing patterns.
    """

    return distribution.sort_values(ascending=ascending).head(n)


def plot_diversity_histogram(distribution: pd.Series, bins: int = 56) -> go.Figure:
    """
    Plot a histogram of product diversity per order.

    This function visualises the distribution of product diversity (number of
    unique products per invoice) as a histogram using Plotly. It provides a
    clear view of how frequently different levels of diversity occur across
    orders.

    Parameters
    ----------
    distribution : pd.Series
        Series containing product diversity values per invoice.
    bins : int, optional
        Number of bins in the histogram (default: 56).

    Returns
    -------
    go.Figure
        Plotly histogram figure showing the distribution of product diversity
        across invoices.

    Notes
    -----
    - Bars are styled with borders for improved readability.
    - Layout is standardised using `apply_common_layout` for consistency.
    - X-axis ticks are adjusted to linear mode for clarity.
    - Useful for identifying common diversity ranges and spotting unusual
      purchasing patterns.
    """

    # Create histogram with Plotly Express
    fig = px.histogram(
        distribution,
        nbins=bins,
        labels={"value": "Number of Unique Products"},
    )

    # Style histogram bars with borders
    fig.update_traces(
        selector=dict(type="histogram"),
        marker=dict(
            line=dict(color="black", width=0.5)  # border around bars
        )
    )

    # Apply standardised layout using helper
    fig = apply_common_layout(
        fig,
        title="Distribution of Product Diversity per Order",
        xaxis_title="Number of Unique Products",
        yaxis_title="Number of Orders",
        showlegend=False
    )

    # Adjust x-axis ticks for readability
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=100)

    return fig


def plot_country_diversity(df: pd.DataFrame) -> go.Figure:
    """
    Plot average product diversity per order by country.

    This function calculates the average product diversity (number of unique
    products per invoice) for each country and visualises the results as a
    horizontal bar chart using Plotly. It highlights differences in purchasing
    patterns across countries.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'InvoiceNo' : invoice identifiers.
        - 'StockCode' : product codes.
        - 'Country' : country of the transaction.

    Returns
    -------
    go.Figure
        Plotly horizontal bar chart showing average product diversity per
        order by country.

    Notes
    -----
    - Product diversity is computed per invoice using `nunique()` on StockCode.
    - Average diversity is then aggregated by country.
    - Bars are labelled with rounded values for clarity.
    - Layout is standardised with `apply_common_layout` for consistent styling.
    - Y-axis categories are ordered by total diversity in ascending order.
    - Useful for comparing customer behaviour across different regions.
    """

    # Compute average product diversity per country in one pass
    order_diversity = (
        df.groupby("InvoiceNo")["StockCode"].nunique().reset_index(name="ProductDiversity")
    )
    country_diversity = (
        order_diversity.merge(df[["InvoiceNo", "Country"]].drop_duplicates(), on="InvoiceNo")
        .groupby("Country")["ProductDiversity"].mean()
        .sort_values(ascending=False)
    )

    fig = px.bar(
        country_diversity.reset_index(),
        x="ProductDiversity",
        y="Country",
        orientation="h",
        text=country_diversity.round(4),
        color="Country"
    )

    fig.update_traces(textposition="outside")

    # Apply standardised layout
    fig = apply_common_layout(
        fig,
        title="Product Diversity by Country",
        xaxis_title="Average Product Diversity per Order",
        yaxis_title="Country",
        showlegend=False,
        height=864
    )
    fig.update_layout(
        yaxis=dict(showgrid=False)
    )
    fig.update_yaxes(categoryorder="total ascending")

    return fig


# Regional Trends
def get_top_products_by_country(df: pd.DataFrame,
                                description_mode: dict) -> pd.DataFrame:
    """
    Return the top product (by revenue) for each country.

    This function aggregates product-level revenue by country and identifies
    the product with the highest revenue in each country. It attaches the most
    frequent description for each product using a precomputed mapping to avoid
    recomputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Country' : country of the transaction.
        - 'StockCode' : product identifier.
        - 'Revenue' : numeric revenue values.
    description_mode : dict
        Dictionary mapping each StockCode to its most frequent Description
        (title-cased). Typically generated once during data cleaning.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per country, including:
        - 'Country' : country name.
        - 'StockCode' : top product identifier by revenue.
        - 'Revenue' : total revenue of the top product in that country.
        - 'Description' : most frequent description of the product.

    Notes
    -----
    - Revenue is aggregated by country and product before selecting the top
      product per country.
    - The description mapping is passed in as a parameter to ensure consistent
      product naming without recomputation.
    - Useful for identifying leading products in each market and comparing
      product performance across regions.
    """

    # Aggregate revenue by country and product
    agg = df.groupby(["Country", "StockCode"], as_index=False)["Revenue"].sum()

    # Find index of max revenue per country
    idx = agg.groupby("Country")["Revenue"].idxmax()
    top_products = agg.loc[idx].reset_index(drop=True)

    # Map description from description_mode.
    top_products["Description"] = top_products["StockCode"].map(description_mode)

    return top_products


def get_basket_size_by_country(df: pd.DataFrame) -> pd.Series:
    """
    Compute average basket size per country.

    This function calculates the average basket size (total quantity of items
    per order) for each country. It aggregates the total quantity purchased
    and the number of unique orders, then divides to obtain the mean basket
    size per country.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'Country' : country of the transaction.
        - 'Quantity' : number of items purchased.
        - 'InvoiceNo' : invoice identifiers.

    Returns
    -------
    pd.Series
        Series indexed by country with values representing the average basket
        size (mean quantity per order). The Series is named "BasketSize".

    Notes
    -----
    - Basket size is computed as total quantity ÷ number of unique orders.
    - Provides insight into purchasing behaviour across different countries.
    - Useful for comparing order sizes and identifying markets with larger or
      smaller average baskets.
    """

    agg = df.groupby("Country").agg(
        total_quantity=("Quantity", "sum"),
        num_orders=("InvoiceNo", "nunique")
    )
    basket_size = agg["total_quantity"] / agg["num_orders"]
    basket_size.name = "BasketSize"
    return basket_size


def plot_basket_size_by_country(
    basket_size_by_country: pd.Series,
    sort_alphabetically: bool = False
) -> go.Figure:
    """
    Plot average basket size per country.

    This function visualises the average basket size (mean product quantity
    per order) across countries as a horizontal bar chart using Plotly. It
    provides a clear comparison of purchasing behaviour between regions.

    Parameters
    ----------
    basket_size_by_country : pd.Series
        Series indexed by country with average basket size values. The Series
        should be named "BasketSize".
    sort_alphabetically : bool, optional
        If True, countries are sorted alphabetically. If False (default),
        countries are sorted by basket size so that the largest values appear
        at the top.

    Returns
    -------
    go.Figure
        Plotly horizontal bar chart showing average basket size per country.

    Notes
    -----
    - Bars are labelled with rounded basket size values for readability.
    - Ordering of countries can be controlled via the `sort_alphabetically`
      flag.
    - Layout is standardised with `apply_common_layout` for consistent styling.
    - Useful for identifying markets with larger or smaller average order sizes.
    """

    # Build bar chart with Plotly Express
    fig = px.bar(
        basket_size_by_country.reset_index(),
        x="BasketSize" if basket_size_by_country.name else basket_size_by_country.reset_index().columns[1],
        y="Country",
        orientation="h",
        text=basket_size_by_country.round(2),
        color="Country"
    )

    # Position text labels outside bars
    fig.update_traces(textposition="outside")

    # Control ordering
    if sort_alphabetically:
        fig.update_yaxes(categoryorder="category descending")
    else:
        fig.update_yaxes(categoryorder="total ascending")

    # Apply standardised layout
    fig = apply_common_layout(
        fig,
        title="Average Product Quantity per Order divided by Country",
        xaxis_title="Average Product Quantity per Order",
        yaxis_title="Country",
        showlegend=False,
        height=864
    )

    return fig


# RFM Analysis
# Segmentation function
def segment(rfm):
    """
    Assign customer segment based on RFM score.

    This function maps a three‑digit RFM score (Recency, Frequency, Monetary)
    to a descriptive customer segment. It uses predefined rules to classify
    customers into categories such as "Champions," "Lost Customers," or
    "Big Spenders," depending on their RFM profile.

    Parameters
    ----------
    rfm : pd.Series
        A row from the RFM DataFrame containing at least:
        - 'RFM_Score' : string representation of the combined R, F, and M scores.

    Returns
    -------
    str
        Customer segment label, one of:
        - "Champions"
        - "Lost Customers"
        - "Loyal Customers"
        - "At Risk"
        - "Price Sensitive"
        - "Recent Customers"
        - "Frequent Buyers"
        - "Big Spenders"
        - "Others"

    Notes
    -----
    - Segmentation is rule‑based and depends on specific RFM score patterns.
    - Useful for customer profiling, marketing strategies, and retention analysis.
    """

    score = rfm["RFM_Score"]
    if score == "555":
        return "Champions"
    elif score == "111":
        return "Lost Customers"
    elif all(c in "45" for c in score):
        return "Loyal Customers"
    elif (score[0] in "12") and all(c in "45" for c in [score[1],score[2]]):
        return "At Risk"
    elif all(c in "45" for c in [score[0],score[1]]) and (score[2] in "12"):
        return "Price Sensitive"
    elif score[0] == "5":
        return "Recent Customers"
    elif score[1] == "5":
        return "Frequent Buyers"
    elif score[2] == "5":
        return "Big Spenders"
    else:
        return "Others"


# Function to merge some of the segments for simplification or visualisation, if required.
def simplified_segment(segment):
    """
    Simplify customer segments into broader categories.

    This function maps detailed RFM segments into fewer, high‑level categories
    for easier interpretation or visualization. It reduces the number of
    distinct groups while preserving the essence of customer value.

    Parameters
    ----------
    segment : str
        Detailed customer segment label, such as "Champions" or "Lost Customers".

    Returns
    -------
    str
        Simplified segment label, one of:
        - "High Value"
        - "Needs Attention"
        - "Growth Potential"
        - "High Spend"
        - "Low Value"

    Notes
    -----
    - "Champions" and "Loyal Customers" are grouped as "High Value".
    - "At Risk" and "Lost Customers" are grouped as "Needs Attention".
    - "Frequent Buyers" and "Recent Customers" are grouped as "Growth Potential".
    - "Big Spenders" is mapped to "High Spend".
    - All other segments default to "Low Value".
    - Useful for dashboards, reporting, or when fewer categories improve clarity.
    """

    if segment in ["Champions", "Loyal Customers"]:
        return "High Value"
    elif segment in ["At Risk", "Lost Customers"]:
        return "Needs Attention"
    elif segment in ["Frequent Buyers", "Recent Customers"]:
        return "Growth Potential"
    elif segment == "Big Spenders":
        return "High Spend"
    else:
        return "Low Value"


def compute_rfm(df: pd.DataFrame, simplify: bool = False) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) scores and customer segments.

    This function calculates RFM metrics for each customer, assigns scores
    based on quintiles, and classifies customers into segments using rule‑based
    logic. Optionally, segments can be simplified into broader categories for
    easier interpretation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - 'CustomerID' : unique customer identifiers.
        - 'InvoiceNo' : invoice identifiers.
        - 'Revenue' : monetary value of each transaction.
        - 'date' : transaction date.
    simplify : bool, optional
        If True, applies simplified segmentation categories. If False (default),
        uses detailed segmentation.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by customer with the following columns:
        - 'Recency' : days since last purchase.
        - 'Frequency' : number of unique invoices.
        - 'Monetary' : total revenue.
        - 'R_Score', 'F_Score', 'M_Score' : quintile scores for each metric.
        - 'RFM_Score' : combined three‑digit score string.
        - 'Segment' : customer segment label (detailed or simplified).

    Notes
    -----
    - Recency is computed relative to one day after the last transaction date.
    - Frequency is ranked before scoring to ensure unique ordering.
    - Segmentation is applied using the `segment` function; if `simplify=True`,
      the `simplified_segment` mapping is used.
    - Useful for customer profiling, retention strategies, and marketing
      analysis.
    """
    # Reference date is one day after the last transaction
    reference_date = df["date"].max() + pd.Timedelta(days=1)

    # Aggregate RFM metrics in one pass
    rfm = (
        df[df["CustomerID"] != 0]
        .groupby("CustomerID")
        .agg(
            Recency=("date", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    # Rank frequency once
    freq_rank = rfm["Frequency"].rank(method="first")

    # Compute R, F, M scores
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(freq_rank, 5, labels=[1, 2, 3, 4, 5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    # Combine into RFM score string
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    rfm["Segment"] = rfm.apply(segment, axis=1)

    if simplify:
        # Apply simplified segmentation to the Segment column
        rfm["Segment"] = rfm["Segment"].map(simplified_segment)

    return rfm


def compute_segmented_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute segmentation summary of customers and revenue.

    This function aggregates metrics by customer segment from a precomputed
    RFM DataFrame and produces a summary showing customer counts, average
    revenue, total revenue, and relative shares. It provides a high‑level
    view of how different customer groups contribute to overall business
    performance.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Precomputed RFM DataFrame containing at least:
        - 'CustomerID' : unique customer identifiers.
        - 'Monetary' : total revenue per customer.
        - 'Segment' : customer segment label (detailed or simplified).

    Returns
    -------
    pd.DataFrame
        Segmentation summary with one row per segment, including:
        - 'total_customers' : number of customers in the segment.
        - 'mean_revenue' : average revenue per customer.
        - 'total_revenue' : total revenue contributed by the segment.
        - 'customer_share' : proportion of customers in the segment.
        - 'revenue_share' : proportion of revenue contributed by the segment.

    Notes
    -----
    - Assumes segmentation has already been applied to the RFM DataFrame.
    - Results are sorted by total revenue in descending order.
    - Useful for identifying which customer groups drive the most value and
      where retention or growth strategies should be focused.
    """

    # Aggregate metrics by segment
    segmented = (
        rfm_df.groupby("Segment", observed=False)
        .agg(
            total_customers=("CustomerID", "count"),
            mean_revenue=("Monetary", "mean"),
            total_revenue=("Monetary", "sum"),
        )
        .sort_values(by="total_revenue", ascending=False)
    )

    # Compute shares
    segmented["customer_share"] = (
        segmented["total_customers"] / segmented["total_customers"].sum()
    )
    segmented["revenue_share"] = (
        segmented["total_revenue"] / segmented["total_revenue"].sum()
    )

    return segmented


def plot_monetary_by_segment(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Plot average revenue per customer by segment.

    This function aggregates average monetary value (revenue) per segment
    from a precomputed RFM DataFrame and visualises the results as a bar
    chart using Plotly. It highlights differences in spending behaviour
    across customer groups.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Precomputed RFM DataFrame containing at least:
        - 'CustomerID' : unique customer identifiers.
        - 'Monetary' : total revenue per customer.
        - 'Segment' : customer segment label (detailed or simplified).

    Returns
    -------
    go.Figure
        Plotly bar chart showing average revenue per customer by segment.

    Notes
    -----
    - Assumes segmentation has already been applied to the RFM DataFrame.
    - Bars are labelled with average revenue values rounded to two decimals.
    - Layout is standardised with `apply_common_layout` for consistent styling.
    - X‑axis labels are rotated for readability.
    - Useful for identifying which customer segments generate the highest
      average revenue.
    """

    # Aggregate monetary value by segment
    monetary_by_segment = (
        rfm_df.groupby("Segment")["Monetary"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Build bar chart
    fig = px.bar(
        monetary_by_segment,
        x="Segment",
        y="Monetary",
        text=monetary_by_segment["Monetary"].apply(lambda x: f"{x:.2f}"),
        color="Segment"
    )

    # Position text labels outside bars
    fig.update_traces(textposition="outside")

    # Apply standardised layout
    fig = apply_common_layout(
        fig,
        title="Average Revenue per Customer by Segment",
        xaxis_title="Segment",
        yaxis_title="Average Revenue",
        showlegend=False
    )

    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=-45)

    return fig


def plot_rfm_heatmap(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Plot average monetary value by R and F scores as a heatmap.

    This function aggregates average monetary value (revenue) by Recency
    and Frequency scores from a precomputed RFM DataFrame and visualises
    the results as a heatmap using Plotly. It highlights how spending
    behaviour varies across different RFM score combinations.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Precomputed RFM DataFrame containing at least:
        - 'R_Score' : recency quintile score.
        - 'F_Score' : frequency quintile score.
        - 'Monetary' : total revenue per customer.

    Returns
    -------
    go.Figure
        Plotly heatmap showing average monetary value by Recency and
        Frequency scores.

    Notes
    -----
    - Assumes RFM scores have already been computed and provided in
      the DataFrame.
    - Heatmap cells represent the mean monetary value for customers
      within each (R, F) score combination.
    - Layout and styling are applied via `build_heatmap_figure` for
      consistency.
    - Useful for identifying which combinations of recency and frequency
      are associated with higher spending.
    """

    # Pivot table: average monetary value by R and F scores
    rfm_heatmap = (
        rfm_df.groupby(["R_Score", "F_Score"], observed=False)["Monetary"]
        .mean()
        .unstack()
    )

    # Build heatmap using helper
    fig = build_heatmap_figure(
        pivot=rfm_heatmap,
        title="Average Monetary Score by R and F Scores",
        xaxis_title="Frequency Score",
        yaxis_title="Recency Score",
        hovertemplate="Recency %{y}<br>Frequency %{x}<br>Avg Monetary %{z:.2f}",
        texttemplate="%{text:.2f}",
        colorscale="YlGnBu",
        autorange=None
    )

    return fig


if __name__ == "__main__":
    main()