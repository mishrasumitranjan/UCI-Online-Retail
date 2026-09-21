import streamlit as st
import pandas as pd

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_data
from src.eda import *
from src.forecasting import *

# Define Main Function
def main():
    stream()


# Define Additional Functions
@st.cache_data
def get_data(path: str):
    return load_data(path)

@st.cache_data
def clean_data_eda_cached(df: pd.DataFrame):
    return clean_data_eda(df)

@st.cache_data
def country_summary_cached(df: pd.DataFrame):
    return country_summary(df)

@st.cache_data
def customer_summary_cached(df: pd.DataFrame):
    return customer_summary(df)

@st.cache_data
def daywise_summary_cached(df: pd.DataFrame):
    return daywise_summary_all(df)

@st.cache_data
def build_monthly_product_sales_cached(df: pd.DataFrame, description_mode: dict):
    return build_monthly_product_sales(df, description_mode)

@st.cache_data
def build_diversity_distribution_cached(df: pd.DataFrame):
    diversity_distribution = build_diversity_distribution(df)
    diversity_distribution = diversity_distribution.rename("Unique Stock Codes")
    return diversity_distribution

@st.cache_data
def compute_rfm_cached(df: pd.DataFrame, simplify: bool = False) -> pd.DataFrame:
    return compute_rfm(df, simplify)

@st.cache_data
def clean_data_modeling_cached(df: pd.DataFrame):
    return clean_data_modeling(df)


def stream():
    st.title("Retail Forecasting Dashboard")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("[Return to Top](#retail-forecasting-dashboard)")
    if st.sidebar.checkbox("Show EDA"):
        st.sidebar.markdown("[Go to EDA](#exploratory-data-analysis)")
    st.sidebar.markdown("[Go to Top Products](#top-products)")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "parquet", "json"])

    if uploaded_file:
        # Save uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load data using your utility
        df = get_data(uploaded_file.name)

        st.success("Custom dataset loaded successfully!")
    else:
        default_path = Path(__file__).resolve().parent.parent / "data" / "Online Retail.csv"
        df = get_data(str(default_path))
        st.info("Using default dataset. Please upload a dataset to replace it if required.")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # # Sidebar Option
    # st.sidebar.title("Choose Analysis Path")
    # choice = st.sidebar.radio("Select mode:", ["None", "Exploratory Data Analysis", "Forecasting"])
    #
    # if choice == "Exploratory Data Analysis":
    #     st.header("Exploratory Data Analysis")
    #     st.success("EDA picked!")
    #
    # elif choice == "Forecasting":
    #     st.header("Forecasting")
    #     st.success("Forecasting picked!")
    #
    # else:
    #     st.info("Please select an option from the sidebar to continue.")

    # # Tab Option
    # tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Forecasting"])
    #
    # with tab1:
    #     if st.button("Run EDA"):
    #         st.write(df.describe())
    #         st.bar_chart(df["Country"].value_counts())
    #         st.subheader("Sales by Country")
    #         with st.expander("Options for this graph"):
    #             countries = st.multiselect("Select countries", df["Country"].unique())
    #
    # with tab2:
    #     if st.button("Run Forecasting"):
    #         st.success("Forecasting picked!")

    # Custom header and anchor example
    # st.markdown("<h4 id='custom-anchor'>Detailed Analysis</h4>", unsafe_allow_html=True)

    # Storing in session_state
    # if "df" not in st.session_state:
    #     st.session_state.df = load_data("data/Online Retail.csv")

    st.header("Exploratory Data Analysis")
    cleaned_df, description_mode = clean_data_eda_cached(df)

    # # Adding two elements in the same line
    # col1, col2 = st.columns([3, 1])
    # n_rows_cleaned_df = 5
    #
    # with col1:
    #     st.subheader("Cleaned Data Preview")
    #
    # with col2:
    #     n_rows_cleaned_df = st.number_input("Rows", min_value=1, max_value=100, value=5)
    # st.dataframe(cleaned_df.head(5 if n_rows_cleaned_df is None else int(n_rows_cleaned_df)))

    st.subheader("Cleaned Data Preview")
    st.dataframe(cleaned_df)

    price_summary_df = price_summary(cleaned_df)
    st.subheader("Price Summary Per Product")
    st.dataframe(price_summary_df)

    stock_codes = price_summary_df["StockCode"].tolist()

    rsd_list_df = price_summary(cleaned_df)
    st.subheader("Top Products by Relative Standard Deviation in Price")
    st.dataframe(rsd_list_df)

    col1, col2 = st.columns([3, 1])
    selected_code = "85160B"

    with col1:
        st.subheader("Price Distribution for each product")
    with col2:
        selected_code = st.selectbox(
            "Select StockCode",
            options=stock_codes
        )
    price_distribution_plot = plot_price_distribution(cleaned_df, stock_code=str(selected_code))
    st.plotly_chart(price_distribution_plot, use_container_width=True)

    # Country Summary Section
    country_summary_df = country_summary_cached(cleaned_df)
    st.subheader("Summary of all Countries")
    st.dataframe(country_summary_df)

    st.subheader("Top Countries by")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_column_top_countries_df = st.selectbox(
            "Select column",
            options=[col for col in country_summary_df.columns if col != "Country"],
            key="selected_column_top_countries_df"
        )
    with col2:
        n_rows = st.number_input(
            "Top n",
            min_value=1,
            max_value=len(country_summary_df),
            value=10,
            step=1,
            format="%d",
            key="n_rows_top_countries_df"
        )
    with col3:
        is_ascending_top_countries_df = st.checkbox("Ascending", value=False, key="is_ascending_top_countries_df")
    n_rows = int(n_rows) if n_rows is not None else 10
    top_countries_df = top_countries_by(country_summary_df,
                                        column = str(selected_column_top_countries_df),
                                        n = n_rows,
                                        ascending = is_ascending_top_countries_df
                                        )
    st.dataframe(top_countries_df)

    st.subheader("Plot Top Countries by")
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_column_top_countries_plot = st.selectbox(
            "Select column",
            options = [col for col in country_summary_df.columns if col != "Country"],
            key = "selected_column_top_countries_plot"
        )
    with col2:
        excluded_countries_top_countries_plot = st.multiselect(
            "Excluded Countries",
            options = country_summary_df["Country"].unique(),
            key = "excluded_countries_top_countries_plot"
        )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        n_rows_top_countries_plot = st.number_input(
            "Top n",
            min_value = 1,
            max_value = len(country_summary_df),
            value = 10,
            step = 1,
            format = "%d",
            key = "n_rows_top_countries_plot"
        )
    with col2:
        is_ascending_top_countries_plot = st.checkbox("Ascending", value=False, key="is_ascending_top_countries_plot")
    with col3:
        is_log_scale_top_counries_plot = st.checkbox("Log scale", value=False, key="is_log_scale_top_counries_plot")
    n_rows_top_countries_plot = int(n_rows_top_countries_plot) if n_rows_top_countries_plot is not None else 10
    top_countries_plot = plot_top_countries(
        country_summary_df,
        column = str(selected_column_top_countries_plot),
        n = n_rows_top_countries_plot,
        ascending = is_ascending_top_countries_plot,
        exclude = excluded_countries_top_countries_plot,
        log_scale = is_log_scale_top_counries_plot
    )
    st.plotly_chart(top_countries_plot, use_container_width=True)

    st.subheader("Treemap of all Countries")
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_column_countries_treemap = st.selectbox(
            "Select column",
            options = [col for col in country_summary_df.columns if col != "Country"],
            key = "selected_column_countries_treemap"
        )
    with col2:
        excluded_countries_countries_treemap = st.multiselect(
            "Excluded Countries",
            options = country_summary_df["Country"].unique(),
            key = "excluded_countries_countries_treemap"
        )
    country_treemap_plot = plot_country_treemap(
        country_summary_df,
        column=str(selected_column_countries_treemap),
        exclude=excluded_countries_countries_treemap
    )
    st.plotly_chart(country_treemap_plot)

    # Customer Summary Section
    customer_summary_df = customer_summary_cached(cleaned_df)
    st.subheader("Summary of all Customers")
    st.dataframe(customer_summary_df)

    st.subheader("Top Customers by")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_column_top_customers_df = st.selectbox(
            "Select column",
            options=customer_summary_df.columns,
            key="selected_column_top_customers_df"
        )
    with col2:
        n_rows = st.number_input(
            "Top n",
            min_value=1,
            max_value=len(customer_summary_df),
            value=10,
            step=1,
            format="%d",
            key="n_rows_top_customers_df"
        )
    with col3:
        is_ascending_top_customers_df = st.checkbox("Ascending", value=False, key="is_ascending_top_customers_df")
    n_rows = int(n_rows) if n_rows is not None else 10
    top_customers_df = top_customers_by(customer_summary_df,
                                        column = str(selected_column_top_customers_df),
                                        n = n_rows,
                                        ascending = is_ascending_top_customers_df
                                        )
    st.dataframe(top_customers_df)

    st.subheader("Filter Customers Transactions")
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        sort_by_column_filter_customers_plot = st.selectbox(
            "Sort by column",
            options = [col for col in customer_summary_df.columns if col != "CustomerID"],
            key = "sort_by_column_filter_customers_plot"
        )
    with col2:
        n_transactions_filter_customers = st.number_input(
            "Transactions",
            min_value=1,
            value=1,
            key="n_transactions_filter_customers"
        )
    with col3:
        n_customers_filter_customers = st.number_input(
            "Customers",
            value=10,
            key="n_customers_filter_customers"
        )
    with col4:
        is_ascending_filter_customers = st.checkbox("Ascending", value=False, key="is_ascending_filter_customers")
    n_transactions_filter_customers = int(
        n_transactions_filter_customers
    ) if n_transactions_filter_customers is not None else 1
    n_customers_filter_customers = int(
        n_customers_filter_customers
    ) if n_customers_filter_customers is not None else 10

    customer_transaction_filter = filter_customers_by_transactions(
        customer_summary_df,
        transactions=n_transactions_filter_customers,
        sort_by=str(sort_by_column_filter_customers_plot),
        ascending=is_ascending_filter_customers,
        n = n_customers_filter_customers
    )
    st.dataframe(customer_transaction_filter)

    st.subheader("Customer Distribution by Country")
    col1, col2 = st.columns([3, 1])
    with col1:
        excluded_countries_customer_distribution = st.multiselect(
            "Excluded Countries",
            options = country_summary_df["Country"].unique(),
            key = "excluded_countries_customer_distribution"
        )
    with col2:
        n_countries_customer_distribution = st.number_input(
            "Countries",
            min_value=0,
            max_value=len(country_summary_df),
            value=10,
            key="n_countries_customer_distribution"
        )
    n_countries_customer_distribution = int(
        n_countries_customer_distribution
    ) if n_countries_customer_distribution is not None else None
    customer_distribution_df = customer_distribution_by_country(
        cleaned_df,
        n = None if n_countries_customer_distribution == 0 else n_countries_customer_distribution,
        exclude = excluded_countries_customer_distribution
    )
    st.dataframe(customer_distribution_df)

    st.subheader("Plotting Customer Distribution by Country")
    col1, col2 = st.columns([3, 1])
    with col1:
        excluded_countries_customer_distribution_plot = st.multiselect(
            "Excluded Countries",
            options = country_summary_df["Country"].unique(),
            key = "excluded_countries_customer_distribution_plot"
        )
    with col2:
        n_countries_customer_distribution_plot = st.number_input(
            "Countries",
            min_value=0,
            max_value=len(country_summary_df),
            value=10,
            key="n_countries_customer_distribution_plot"
        )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        plot_orientation_customer_distribution_plot = st.selectbox(
            "Orientation",
            options=["Vertical", "Horizontal"],
            key="plot_orientation_customer_distribution_plot"
        )
    with col2:
        is_ascending_customer_distribution_plot = st.checkbox(
            "Ascending",
            value=False,
            key="is_ascending_customer_distribution_plot"
        )
    with col3:
        is_log_scale_customer_distribution_plot = st.checkbox(
            "Log Scale",
            value=False,
            key="is_log_scaleis_log_scale_customer_distribution_plot"
        )
    plot_orientation_cdp = "h" if plot_orientation_customer_distribution_plot == "Horizontal" else "v"
    n_countries_customer_distribution_plot = int(
        n_countries_customer_distribution_plot
    ) if n_countries_customer_distribution_plot is not None else None
    customer_distribution_by_country_plot = plot_customer_distribution_by_country(
        cleaned_df,
        n=n_countries_customer_distribution_plot,
        ascending=is_ascending_customer_distribution_plot,
        exclude=excluded_countries_customer_distribution_plot,
        orientation=plot_orientation_cdp,
        log_scale=is_log_scale_customer_distribution_plot
    )
    st.plotly_chart(customer_distribution_by_country_plot)

    # Day-wise Details Section
    st.subheader("Daily Summary")
    daywise_summary_df = daywise_summary_cached(cleaned_df)
    st.dataframe(daywise_summary_df)

    st.subheader("Plotting Daily Distribution of Metrics")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_column_daywise_metrics_plot = st.selectbox(
            "Select column",
            options = [col for col in daywise_summary_df.columns if col != "date"],
            key = "selected_column_daywise_metrics_plot"
        )
    with col2:
        show_gaps_daywise_metrics_plot = st.checkbox(
            "Show Gaps",
            value=True,
            key="show_gaps_daywise_metrics_plot"
        )
    daywise_metric_plot = plot_daywise_metric_with_gaps(
        daywise_summary_df,
        column=selected_column_daywise_metrics_plot,
        show_gaps=show_gaps_daywise_metrics_plot
    )
    st.plotly_chart(daywise_metric_plot)

    st.subheader("Plotting Average Daily Revenue")
    st.plotly_chart(plot_average_daily_revenue(daywise_summary_df))

    st.subheader("Plotting Various Timely Averages of Daily Revenue")
    st.plotly_chart(plot_timely_avg_revenue(daywise_summary_df))

    st.subheader("Plotting Histogram of Daily Total Revenue")
    bins_revenue_histogram = st.number_input(
        "Bin Size",
        min_value=100,
        value=5000,
        step=100,
        format="%d",
        key="bins_revenue_histogram"
    )
    bins_revenue_histogram = int(bins_revenue_histogram) if bins_revenue_histogram is not None else 5000
    st.plotly_chart(plot_revenue_histogram(daywise_summary_df, bin_size=bins_revenue_histogram))

    st.subheader("Plotting Average Daily Quantity")
    st.plotly_chart(plot_average_daily_quantity(daywise_summary_df))

    st.subheader("Plotting Various Timely Averages of Daily Quantity")
    st.plotly_chart(plot_timely_avg_quantity(daywise_summary_df))

    st.subheader("Plotting Histogram of Daily Total Quantity")
    bins_quantity_histogram = st.number_input(
        "Bin Size",
        min_value=100,
        value=2500,
        step=100,
        format="%d",
        key="bins_quantity_histogram"
    )
    bins_quantity_histogram = int(bins_quantity_histogram) if bins_quantity_histogram is not None else 2500
    st.plotly_chart(plot_quantity_histogram(daywise_summary_df, bin_size=bins_quantity_histogram))

    st.subheader("Plotting Revenue Heatmap by Weekday and Hour")
    st.plotly_chart(plot_revenue_heatmap_by_weekday_hour(cleaned_df))

    st.subheader("Plotting Revenue Heatmap by Weekday and Hour")
    st.plotly_chart(plot_quantity_heatmap_by_weekday_hour(cleaned_df))

    st.subheader("Plotting Revenue Heatmap by Weekday and Month")
    st.plotly_chart(plot_revenue_heatmap_by_weekday_month(cleaned_df))

    st.subheader("Plotting Revenue Heatmap by Weekday and Month")
    st.plotly_chart(plot_quantity_heatmap_by_weekday_month(cleaned_df))

    st.subheader("Plotting Revenue vs. Quantity Scattermap")
    st.plotly_chart(plot_revenue_vs_quantity_scatter(daywise_summary_df))

    st.subheader("Comparing Revenue vs. Quantity")
    st.plotly_chart(plot_revenue_and_quantity_over_time(daywise_summary_df))

    st.subheader("Plotting Heatmap of Average Revenue per Unit Sold by Weekday and Month")
    st.plotly_chart(plot_metric_heatmap_by_weekday_month_premium(cleaned_df))

    st.subheader("Plotting Heatmap of Average Revenue per Unit Sold by Weekday and Hour")
    st.plotly_chart(plot_metric_heatmap_by_weekday_hour_premium(cleaned_df))

    st.subheader("Plotting Revenue Boxplot by Weekday")
    plot_orientation_revenue_boxplot = st.selectbox(
        "Orientation",
        options=["Vertical", "Horizontal"],
        key="plot_orientation_revenue_boxplot"
    )
    plot_orientation_revenue_boxplot = "h" if plot_orientation_customer_distribution_plot == "Horizontal" else "v"
    st.plotly_chart(plot_revenue_boxplot_by_weekday(cleaned_df, orientation=plot_orientation_revenue_boxplot))

    time_cat_options = ["day_of_week", "hour", "day", "week", "month", "quarter", "year"]
    all_cat_options = time_cat_options + ["Country"]

    st.subheader("Comparing Average Revenue vs. Average Quantity")
    comparison_column_revenue_vs_quantity_bar = st.selectbox(
        "Compare by column",
        options=all_cat_options,
        key="comparison_column_revenue_vs_quantity_bar"
    )
    st.plotly_chart(plot_avg_metrics_comparison(cleaned_df, group_col=comparison_column_revenue_vs_quantity_bar))

    st.subheader("Plotting Orders Placed")
    group_by_column_order_time_bar = st.selectbox(
        "Group by",
        options=time_cat_options,
        key="group_by_column_order_time_bar"
    )
    st.plotly_chart(plot_orders_by_time(cleaned_df, group_col=group_by_column_order_time_bar))

    st.subheader("Plotting Heatmap of Orders Placed")
    col1, col2 = st.columns([1, 1])
    with col1:
        row_orders_heatmap = st.selectbox(
            "Row",
            options=[opt for opt in all_cat_options if opt != st.session_state.get("col_orders_heatmap")],
            key="row_orders_heatmap"
        )
    with col2:
        col_orders_heatmap = st.selectbox(
            "Column",
            options=[opt for opt in all_cat_options if opt != row_orders_heatmap],
            key="col_orders_heatmap"
        )
    st.plotly_chart(
        plot_orders_heatmap(cleaned_df, row_col=row_orders_heatmap, col_col=col_orders_heatmap)
    )

    st.subheader("Plotting Heatmap of Revenue per Order")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        row_rpo_heatmap = st.selectbox(
            "Row",
            options=[opt for opt in all_cat_options if opt != st.session_state.get("col_rpo_heatmap")],
            key="row_rpo_heatmap"
        )
    with col2:
        col_rpo_heatmap = st.selectbox(
            "Column",
            options=[opt for opt in all_cat_options if opt != row_rpo_heatmap],
            key="col_rpo_heatmap"
        )
    with col3:
        exclude_single_orders_rpo_heatmap = st.checkbox(
            "Exclude Single Orders",
            value=False,
            key="exclude_single_orders_rpo_heatmap"
        )
    st.plotly_chart(
        plot_rpo_heatmap(
            cleaned_df,
            row_col=row_rpo_heatmap,
            col_col=col_rpo_heatmap,
            exclude_single_orders=exclude_single_orders_rpo_heatmap
        )
    )

    st.subheader("Plotting Heatmap of Quantity per Order")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        row_qpo_heatmap = st.selectbox(
            "Row",
            options=[opt for opt in all_cat_options if opt != st.session_state.get("col_qpo_heatmap")],
            key="row_qpo_heatmap"
        )
    with col2:
        col_qpo_heatmap = st.selectbox(
            "Column",
            options=[opt for opt in all_cat_options if opt != row_qpo_heatmap],
            key="col_qpo_heatmap"
        )
    with col3:
        exclude_single_orders_qpo_heatmap = st.checkbox(
            "Exclude Single Orders",
            value=False,
            key="exclude_single_orders_qpo_heatmap"
        )
    st.plotly_chart(
        plot_qpo_heatmap(
            cleaned_df,
            row_col=row_qpo_heatmap,
            col_col=col_qpo_heatmap,
            exclude_single_orders=exclude_single_orders_qpo_heatmap
        )
    )

    st.subheader("Top Products by")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_column_top_products_df = st.selectbox(
            "Select column",
            options=["Revenue", "Quantity", "Orders"],
            key="selected_column_top_products_df"
        )
    with col2:
        n_rows_top_products_df = st.number_input(
            "Top n",
            min_value=1,
            max_value=len(price_summary_df),
            value=10,
            step=1,
            format="%d",
            key="n_rows_top_products_df"
        )
    with col3:
        is_ascending_top_products_df = st.checkbox("Ascending", value=False, key="is_ascending_top_products_df")
    n_rows_top_products_df = int(n_rows_top_products_df) if n_rows_top_products_df is not None else 10
    top_products_df = get_top_bottom_products(
        cleaned_df,
        sort_col = str(selected_column_top_products_df),
        n = n_rows_top_products_df,
        ascending = is_ascending_top_products_df
    )
    st.dataframe(top_products_df)

    st.subheader("Monthly Product Sales (Preview)")
    monthly_product_sales = build_monthly_product_sales_cached(cleaned_df, description_mode)
    st.dataframe(monthly_product_sales)

    st.subheader("Products with Highest Sales Each Month")
    n_rows_top_products_by_month = st.number_input(
        "Top n",
        min_value=1,
        max_value=len(price_summary_df),
        value=1,
        step=1,
        format="%d",
        key="n_rows_top_products_by_month"
    )
    n_rows_top_products_by_month = int(
        n_rows_top_products_by_month
    ) if n_rows_top_products_by_month is not None else 1
    top_products_by_month_df = top_products_by_month(monthly_product_sales, n=n_rows_top_products_by_month)
    st.dataframe(top_products_by_month_df)

    st.subheader("Products sorted by Volatility (Relative Standard Deviation) in Price")
    col1, col2 = st.columns([1, 1])
    with col1:
        n_rows_top_volatile_products_by_rsd = st.number_input(
            "Top n",
            min_value=1,
            max_value=len(price_summary_df),
            value=10,
            step=1,
            format="%d",
            key="n_rows_top_volatile_products_by_rsd"
        )
    with col2:
        is_ascending_top_volatile_products_by_rsd = st.checkbox(
            "Ascending",
            value=False,
            key="is_ascending_top_volatile_products_by_rsd"
        )
    n_rows_top_volatile_products_by_rsd = int(
        n_rows_top_volatile_products_by_rsd
    ) if n_rows_top_volatile_products_by_rsd is not None else 10
    top_volatile_products_by_rsd_df = top_volatile_products_by_rsd(
        monthly_product_sales,
        description_mode=description_mode,
        n=n_rows_top_volatile_products_by_rsd,
        ascending=is_ascending_top_volatile_products_by_rsd
    )
    st.dataframe(top_volatile_products_by_rsd_df)

    st.subheader("Heatmap of Monthly Revenue")
    n_rows_monthly_revenue_heatmap = st.number_input(
        "Top n",
        min_value=1,
        max_value=len(price_summary_df),
        value=10,
        step=1,
        format="%d",
        key="n_rows_monthly_revenue_heatmap"
    )
    n_rows_monthly_revenue_heatmap = int(
        n_rows_monthly_revenue_heatmap
    ) if n_rows_monthly_revenue_heatmap is not None else 10
    monthly_revenue_heatmap = plot_monthly_revenue_heatmap(monthly_product_sales, n=n_rows_monthly_revenue_heatmap)
    st.plotly_chart(monthly_revenue_heatmap)

    st.subheader("Plotting the Monthly Revenue Trend of the Product with Most Sales")
    top_product_monthly_revenue_trend = plot_top_product_monthly_revenue_trend(monthly_product_sales, description_mode)
    st.plotly_chart(top_product_monthly_revenue_trend)

    # Product Diversity
    st.subheader("Product Diversity per Invoice/Order (Preview)")
    diversity_distribution = build_diversity_distribution_cached(cleaned_df)
    st.dataframe(diversity_distribution)

    st.subheader("Invoices sorted by Diversity")
    col1, col2 = st.columns([1, 1])
    with col1:
        n_rows_top_diverse_invoices = st.number_input(
            "Top n",
            min_value=1,
            max_value=len(diversity_distribution),
            value=10,
            step=1,
            format="%d",
            key="n_rows_top_diverse_invoices"
        )
    with col2:
        is_ascending_top_diverse_invoices = st.checkbox(
            "Ascending",
            value=False,
            key="is_ascending_top_diverse_invoices"
        )
    n_rows_top_diverse_invoices = int(
        n_rows_top_diverse_invoices
    ) if n_rows_top_diverse_invoices is not None else 10
    top_diverse_invoices = get_top_diverse_invoices(
        diversity_distribution,
        n=n_rows_top_diverse_invoices,
        ascending=is_ascending_top_diverse_invoices
    )
    st.dataframe(top_diverse_invoices)

    st.subheader("Product Distribution Histogram")
    bins_diversity_histogram = st.number_input(
        "Bins",
        min_value=1,
        max_value=len(diversity_distribution),
        value=56,
        step=1,
        format="%d",
        key="bins_diversity_histogram"
    )
    bins_diversity_histogram = int(
        bins_diversity_histogram
    ) if bins_diversity_histogram is not None else 56
    diversity_histogram = plot_diversity_histogram(diversity_distribution, bins=bins_diversity_histogram)
    st.plotly_chart(diversity_histogram)

    st.subheader("Plotting Average Product Diversity per Order by Country")
    st.plotly_chart(plot_country_diversity(cleaned_df))

    st.subheader("Top Product by Revenue for each Country")
    st.dataframe(get_top_products_by_country(cleaned_df, description_mode))

    st.subheader("Average Quantity of Items per Order for each Country (Preview)")
    avg_basket_size_by_country = get_basket_size_by_country(cleaned_df)
    st.dataframe(avg_basket_size_by_country)

    st.subheader("Plotting Average Quantity of Items per Order for each Country")
    sort_alphabetically_plot_basket_size_by_country = st.checkbox(
        "Sort Alphabetically",
        value=False,
        key="sort_alphabetically_plot_basket_size_by_country"
    )
    st.plotly_chart(
        plot_basket_size_by_country(
            avg_basket_size_by_country,
            sort_alphabetically=sort_alphabetically_plot_basket_size_by_country
        )
    )

    st.header("RFM Analysis")
    st.subheader("Calculating RFM (Preview)")
    simplify_categories_compute_rfm = st.checkbox(
        "Simplify Categories",
        value=False,
        key="simplify_categories_compute_rfm"
    )
    rfm_df = compute_rfm_cached(cleaned_df, simplify=simplify_categories_compute_rfm)
    st.dataframe(rfm_df)

    st.subheader("Summary of all Segments")
    st.dataframe(compute_segmented_summary(rfm_df))

    st.subheader("Plotting Monetary Score by Segment")
    st.plotly_chart(plot_monetary_by_segment(rfm_df))

    st.subheader("Plotting Heatmap of Average Monetary Score by Recency and Frequency Scores")
    st.plotly_chart(plot_rfm_heatmap(rfm_df))

    st.header("Forecasting")
    st.subheader("Processed Data for Forecasting (Preview)")
    cleaned_df_modeling = clean_data_modeling_cached(df)
    st.dataframe(cleaned_df_modeling)

    rfp = RevenueForecastPipeline()
    rfp.preprocess(df)
    rfp.fit()
    rfp.predict()

    st.subheader("Plotting the Results of the Modeling")
    st.plotly_chart(rfp.graph_blended_results())
    evaluation_results = rfp.evaluate()
    st.table(pd.DataFrame(evaluation_results.items(), columns=["Metric", "Value"]))

    st.subheader("Forecasting")
    col1, col2 = st.columns([1, 1])
    with col1:
        forecast_history_days = st.number_input(
            "Past Days",
            min_value=1,
            max_value=len(daywise_summary_df),
            value=7,
            step=1,
            format="%d",
            key="forecast_history_days"
        )
    with col2:
        forecast_predicted_days = st.number_input(
            "Predict Days",
            min_value=1,
            max_value=31,
            value=7,
            step=1,
            format="%d",
            key="forecast_predicted_days"
        )
    forecast_history_days = int(
        forecast_history_days
    ) if forecast_history_days is not None else 7
    forecast_predicted_days = int(
        forecast_predicted_days
    ) if forecast_predicted_days is not None else 7
    forecasted_df = rfp.forecast(forecast_predicted_days)
    st.dataframe(forecasted_df)
    forecasted_graph = rfp.graph_forecasted_results(
        history_days=forecast_history_days,
        forecast_days=forecast_predicted_days
    )
    st.plotly_chart(forecasted_graph)

#----



if __name__ == "__main__":
    main()

