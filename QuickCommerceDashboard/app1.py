"""
Quick-Commerce Delivery and Customer Experience Analytics Dashboard
Main Application: app1.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quick-Commerce Delivery Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive, Theme-Aware Styling (Optimized for both Light and Dark themes)
st.markdown("""
<style>
    /* Main container spacing */
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }
    
    /* Clean typography inheriting theme text colors */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        color: var(--text-color);
        letter-spacing: -0.02em;
    }
    
    /* Dashboard Header styling */
    .dashboard-header {
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .dashboard-subtitle {
        color: var(--text-color);
        opacity: 0.75;
        font-size: 0.98rem;
        margin-top: -0.25rem;
    }
    
    /* Modern Theme-Aware KPI Metric Cards */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
        border-color: rgba(128, 128, 128, 0.35);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-color) !important;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
    }
    
    /* Insight Cards */
    .insight-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: var(--text-color);
    }
    .insight-card strong {
        color: var(--text-color);
    }
    
    /* Plotly Chart Container */
    div[data-testid="stPlotlyChart"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
    }
    
    /* Interpretation box */
    .chart-caption {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.85;
        margin-top: -6px;
        margin-bottom: 18px;
    }
    
    /* Sidebar styling: border only so background adapts naturally to light & dark */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    .sidebar-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background-color: rgba(37, 99, 235, 0.15);
        color: #3B82F6;
        border: 1px solid rgba(37, 99, 235, 0.3);
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA LOADING & PRE-PROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads and standardizes the quick-commerce delivery dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "ecommerce_delivery_analytics.csv")
    if not os.path.exists(file_path):
        file_path = "ecommerce_delivery_analytics.csv"

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Numerical Column Normalization
    numeric_columns = [
        "Delivery Time (Minutes)",
        "Order Value (INR)",
        "Service Rating"
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False).str.strip(),
                errors="coerce"
            )

    # Categorical Column Normalization
    categorical_columns = [
        "Platform",
        "Product Category",
        "Delivery Delay",
        "Refund Requested",
        "Customer Feedback"
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    return df


df = load_data()

# Column Names Constants
platform_col = "Platform"
category_col = "Product Category"
order_col = "Order ID"
delivery_col = "Delivery Time (Minutes)"
value_col = "Order Value (INR)"
rating_col = "Service Rating"
delay_col = "Delivery Delay"
refund_col = "Refund Requested"

required_columns = [
    platform_col, category_col, order_col,
    delivery_col, value_col, rating_col,
    delay_col, refund_col
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing required columns in dataset: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()


# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("Dashboard Filters")
st.sidebar.caption("Filter data dynamically across all tabs and metrics.")

all_platforms = sorted(df[platform_col].unique().tolist())
all_categories = sorted(df[category_col].unique().tolist())

# Numeric bounds
min_val = float(df[value_col].min())
max_val = float(df[value_col].max())
min_del = int(df[delivery_col].min())
max_del = int(df[delivery_col].max())

# Reset filter callback
def reset_filter_state():
    st.session_state["filter_platforms"] = all_platforms
    st.session_state["filter_categories"] = all_categories
    st.session_state["filter_delay"] = "All"
    st.session_state["filter_val_range"] = (min_val, max_val)
    st.session_state["filter_del_range"] = (min_del, max_del)
    st.session_state["filter_min_rating"] = 1.0

# Initialize session state if not set
if "filter_platforms" not in st.session_state:
    reset_filter_state()

# Interactive Filter Widgets
selected_platforms = st.sidebar.multiselect(
    "Delivery Platform",
    options=all_platforms,
    key="filter_platforms"
)

selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=all_categories,
    key="filter_categories"
)

delay_option = st.sidebar.selectbox(
    "Delivery Delay Status",
    options=["All", "Delayed Only (Yes)", "On-Time Only (No)"],
    key="filter_delay"
)

selected_val_range = st.sidebar.slider(
    "Order Value Range (INR)",
    min_value=min_val,
    max_value=max_val,
    step=50.0,
    format="₹%d",
    key="filter_val_range"
)

selected_del_range = st.sidebar.slider(
    "Delivery Time Range (Minutes)",
    min_value=min_del,
    max_value=max_del,
    step=1,
    format="%d min",
    key="filter_del_range"
)

selected_min_rating = st.sidebar.slider(
    "Minimum Customer Rating",
    min_value=1.0,
    max_value=5.0,
    step=0.5,
    format="%.1f ⭐",
    key="filter_min_rating"
)

st.sidebar.button("Reset Filters", on_click=reset_filter_state, use_container_width=True)

# Apply Filter Mask
mask = (
    (df[platform_col].isin(selected_platforms)) &
    (df[category_col].isin(selected_categories)) &
    (df[value_col].between(selected_val_range[0], selected_val_range[1])) &
    (df[delivery_col].between(selected_del_range[0], selected_del_range[1])) &
    (df[rating_col] >= selected_min_rating)
)

if delay_option == "Delayed Only (Yes)":
    mask = mask & (df[delay_col] == "Yes")
elif delay_option == "On-Time Only (No)":
    mask = mask & (df[delay_col] == "No")

filtered_df = df[mask].copy()

# Sidebar summary badge
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div class='sidebar-badge'>Active Records: {len(filtered_df):,} / {len(df):,} "
    f"({(len(filtered_df)/len(df)*100 if len(df) > 0 else 0):.1f}%)</div>",
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 4. DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="dashboard-header">
    <h1 style="margin-bottom: 2px;">Quick-Commerce Delivery & Customer Experience Analytics</h1>
    <div class="dashboard-subtitle">
        Performance monitoring, customer satisfaction insights, and operational analytics across quick-commerce platforms.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. FOUR APPLICATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard Overview",
    "Dataset Overview",
    "Data Analysis",
    "Visualizations"
])


# =============================================================================
# TAB 1: DASHBOARD OVERVIEW
# =============================================================================
with tab1:
    if filtered_df.empty:
        st.warning("No records match the active filter criteria. Please adjust your sidebar filters or click 'Reset Filters'.")
    else:
        # Key KPI Calculations
        total_orders = len(filtered_df)
        total_revenue = filtered_df[value_col].sum()
        avg_order_value = filtered_df[value_col].mean()
        avg_delivery_time = filtered_df[delivery_col].mean()
        avg_rating = filtered_df[rating_col].mean()
        delayed_orders_count = (filtered_df[delay_col] == "Yes").sum()
        delayed_orders_rate = (delayed_orders_count / total_orders * 100) if total_orders > 0 else 0.0

        # Display 6 Metric Cards in 6 Columns
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

        kpi1.metric("Total Orders", f"{total_orders:,}")
        kpi2.metric("Total Order Value", f"₹{total_revenue:,.0f}")
        kpi3.metric("Avg Order Value", f"₹{avg_order_value:,.2f}")
        kpi4.metric("Avg Delivery Time", f"{avg_delivery_time:.1f} mins")
        kpi5.metric("Avg Rating", f"{avg_rating:.2f} / 5.0 ⭐")
        kpi6.metric(
            "Delayed Orders",
            f"{delayed_orders_count:,}",
            delta=f"{delayed_orders_rate:.1f}% rate",
            delta_color="inverse"
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Overview Visualizations: 2 Columns
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.subheader("Platform-wise Order Share")
            platform_counts = (
                filtered_df.groupby(platform_col)
                .size()
                .reset_index(name="Orders")
            )
            fig_donut = px.pie(
                platform_counts,
                names=platform_col,
                values="Orders",
                hole=0.55,
                color_discrete_sequence=["#2563EB", "#0D9488", "#F59E0B"]
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Orders: %{value:,}<br>Share: %{percent}<extra></extra>"
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Displays order volume distribution across platforms. "
                "Highlights market share and platform preference in the quick-commerce segment.</div>",
                unsafe_allow_html=True
            )

        with chart_col2:
            st.subheader("Orders by Product Category")
            category_counts = (
                filtered_df.groupby(category_col)
                .size()
                .reset_index(name="Orders")
                .sort_values("Orders", ascending=True)
            )
            fig_cat = px.bar(
                category_counts,
                x="Orders",
                y=category_col,
                orientation="h",
                text="Orders",
                color=category_col,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_cat.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Total Orders: %{x:,}<extra></extra>"
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=40),
                xaxis_title="Number of Orders",
                yaxis_title=""
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Compares order frequencies across product categories. "
                "Identifies highest-demand consumer categories in quick delivery.</div>",
                unsafe_allow_html=True
            )

        # Executive Insights Section
        st.subheader("Key Operational Highlights")
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)

        # Calculate highlight values safely
        top_platform = filtered_df[platform_col].value_counts().idxmax() if not filtered_df.empty else "N/A"
        fastest_platform = (
            filtered_df.groupby(platform_col)[delivery_col].mean().idxmin()
            if not filtered_df.empty else "N/A"
        )
        fastest_time = (
            filtered_df.groupby(platform_col)[delivery_col].mean().min()
            if not filtered_df.empty else 0.0
        )
        top_category_rev = (
            filtered_df.groupby(category_col)[value_col].sum().idxmax()
            if not filtered_df.empty else "N/A"
        )
        refund_count = (filtered_df[refund_col] == "Yes").sum()
        refund_pct = (refund_count / total_orders * 100) if total_orders > 0 else 0.0

        with h_col1:
            st.markdown(
                f"<div class='insight-card'><strong>Top Volume Platform</strong><br>"
                f"<span style='font-size: 1.15rem; font-weight: 600; color: #3B82F6;'>{top_platform}</span><br>"
                f"<span style='color: var(--text-color); opacity: 0.75; font-size: 0.85rem;'>Largest share of total orders</span></div>",
                unsafe_allow_html=True
            )

        with h_col2:
            st.markdown(
                f"<div class='insight-card' style='border-left-color: #10B981;'><strong>Fastest Delivery Platform</strong><br>"
                f"<span style='font-size: 1.15rem; font-weight: 600; color: #10B981;'>{fastest_platform}</span><br>"
                f"<span style='color: var(--text-color); opacity: 0.75; font-size: 0.85rem;'>Average {fastest_time:.1f} minutes</span></div>",
                unsafe_allow_html=True
            )

        with h_col3:
            st.markdown(
                f"<div class='insight-card' style='border-left-color: #A855F7;'><strong>Top Revenue Category</strong><br>"
                f"<span style='font-size: 1.15rem; font-weight: 600; color: #A855F7;'>{top_category_rev}</span><br>"
                f"<span style='color: var(--text-color); opacity: 0.75; font-size: 0.85rem;'>Highest gross order value</span></div>",
                unsafe_allow_html=True
            )

        with h_col4:
            st.markdown(
                f"<div class='insight-card' style='border-left-color: #F59E0B;'><strong>Refund Request Rate</strong><br>"
                f"<span style='font-size: 1.15rem; font-weight: 600; color: #F59E0B;'>{refund_pct:.1f}%</span><br>"
                f"<span style='color: var(--text-color); opacity: 0.75; font-size: 0.85rem;'>{refund_count:,} orders requested refund</span></div>",
                unsafe_allow_html=True
            )


# =============================================================================
# TAB 2: DATASET OVERVIEW
# =============================================================================
with tab2:
    st.subheader("Dataset Summary & Structural Properties")
    
    # Dataset Summary Metrics
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.metric("Total Records", f"{len(df):,}")
    meta_col2.metric("Total Attributes", len(df.columns))
    meta_col3.metric("Total Missing Values", int(df.isnull().sum().sum()))
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    meta_col4.metric("Memory Footprint", f"{memory_mb:.2f} MB")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Dataset Preview
    st.subheader("Dataset Preview")
    preview_limit = st.selectbox(
        "Display Rows:",
        options=[10, 25, 50, 100],
        index=0,
        help="Select the number of initial sample rows to view."
    )
    st.dataframe(df.head(preview_limit), use_container_width=True)

    # Column Schema & Metadata Table
    st.subheader("Attribute Metadata and Schema")
    schema_df = pd.DataFrame({
        "Attribute": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Sample Value": [str(df[col].iloc[0]) if len(df) > 0 else "N/A" for col in df.columns]
    })
    st.dataframe(schema_df, use_container_width=True)

    st.divider()

    # DESCRIPTIVE STATISTICS (CLEARLY VISIBLE, NOT HIDDEN IN EXPANDERS)
    st.subheader("Descriptive Statistics (Numerical Attributes)")
    st.caption("Standard 8-point statistical summary: Count, Mean, Standard Deviation, Min, 25%, 50% (Median), 75%, and Max.")

    num_cols = [delivery_col, value_col, rating_col]
    desc_stats = df[num_cols].describe(percentiles=[0.25, 0.50, 0.75]).T
    desc_stats = desc_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc_stats.columns = [
        "Count", "Mean", "Std Deviation", "Min",
        "25% (Q1)", "50% (Median)", "75% (Q3)", "Max"
    ]
    formatted_desc = desc_stats.copy()
    formatted_desc["Count"] = formatted_desc["Count"].apply(lambda x: f"{int(x):,}")
    for c in ["Mean", "Std Deviation", "Min", "25% (Q1)", "50% (Median)", "75% (Q3)", "Max"]:
        formatted_desc[c] = formatted_desc[c].apply(lambda x: f"{x:,.2f}")

    st.dataframe(formatted_desc, use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Categorical Feature Summary
    st.subheader("Categorical Feature Summary")
    cat_cols_list = [platform_col, category_col, delay_col, refund_col]
    cat_summary = df[cat_cols_list].describe().T
    cat_summary.columns = ["Total Count", "Unique Values", "Top Mode", "Mode Frequency"]
    cat_summary["Total Count"] = cat_summary["Total Count"].apply(lambda x: f"{int(x):,}")
    cat_summary["Mode Frequency"] = cat_summary["Mode Frequency"].apply(lambda x: f"{int(x):,}")
    st.dataframe(cat_summary, use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Missing Value Analysis Table
    st.subheader("Missing Value Analysis")
    missing_analysis = pd.DataFrame({
        "Attribute": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage (%)": (df.isnull().mean() * 100).round(2).values,
        "Data Health Status": ["Complete (0% missing)" if m == 0 else f"{m} missing" for m in df.isnull().sum().values]
    })
    st.dataframe(missing_analysis, use_container_width=True)


# =============================================================================
# TAB 3: DATA ANALYSIS
# =============================================================================
with tab3:
    if filtered_df.empty:
        st.warning("No records match the active filter criteria. Please adjust your sidebar filters or click 'Reset Filters'.")
    else:
        st.subheader("1. Platform-wise Comprehensive Performance Analysis")
        
        # Group-by Platform Performance Table
        platform_perf = filtered_df.groupby(platform_col).agg(
            Total_Orders=(order_col, "count"),
            Total_Revenue=(value_col, "sum"),
            Avg_Order_Value=(value_col, "mean"),
            Avg_Delivery_Time=(delivery_col, "mean"),
            Avg_Rating=(rating_col, "mean"),
            Delayed_Orders=(delay_col, lambda s: (s == "Yes").sum()),
            Refund_Orders=(refund_col, lambda s: (s == "Yes").sum())
        ).reset_index()

        total_orders_all = len(filtered_df)
        platform_perf["Order Share (%)"] = (platform_perf["Total_Orders"] / total_orders_all * 100).round(2)
        platform_perf["Delay Rate (%)"] = (platform_perf["Delayed_Orders"] / platform_perf["Total_Orders"] * 100).round(2)
        platform_perf["Refund Rate (%)"] = (platform_perf["Refund_Orders"] / platform_perf["Total_Orders"] * 100).round(2)

        display_platform_perf = platform_perf[[
            platform_col, "Total_Orders", "Order Share (%)", "Total_Revenue",
            "Avg_Order_Value", "Avg_Delivery_Time", "Avg_Rating",
            "Delayed_Orders", "Delay Rate (%)", "Refund Rate (%)"
        ]].copy()

        display_platform_perf.columns = [
            "Platform", "Orders", "Share (%)", "Revenue (INR)",
            "AOV (INR)", "Avg Delivery (Min)", "Avg Rating",
            "Delayed Orders", "Delay Rate (%)", "Refund Rate (%)"
        ]

        st.dataframe(
            display_platform_perf.style.format({
                "Orders": "{:,}",
                "Share (%)": "{:.1f}%",
                "Revenue (INR)": "₹{:,.2f}",
                "AOV (INR)": "₹{:.2f}",
                "Avg Delivery (Min)": "{:.1f} m",
                "Avg Rating": "{:.2f} ⭐",
                "Delayed Orders": "{:,}",
                "Delay Rate (%)": "{:.1f}%",
                "Refund Rate (%)": "{:.1f}%"
            }),
            use_container_width=True
        )

        st.markdown(
            "<div class='chart-caption'><b>Interpretation:</b> Platform comparison reveals operational trade-offs "
            "between delivery speed, revenue throughput, customer rating, and delay rates.</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # Product Category Performance Analysis
        st.subheader("2. Category-wise Performance Summary")
        cat_perf = filtered_df.groupby(category_col).agg(
            Total_Orders=(order_col, "count"),
            Total_Revenue=(value_col, "sum"),
            Avg_Order_Value=(value_col, "mean"),
            Avg_Delivery_Time=(delivery_col, "mean"),
            Avg_Rating=(rating_col, "mean")
        ).reset_index().sort_values("Total_Revenue", ascending=False)

        cat_perf["Revenue Share (%)"] = (cat_perf["Total_Revenue"] / cat_perf["Total_Revenue"].sum() * 100).round(2)

        st.dataframe(
            cat_perf.style.format({
                "Total_Orders": "{:,}",
                "Total_Revenue": "₹{:,.2f}",
                "Avg_Order_Value": "₹{:.2f}",
                "Avg_Delivery_Time": "{:.1f} m",
                "Avg_Rating": "{:.2f} ⭐",
                "Revenue Share (%)": "{:.1f}%"
            }),
            use_container_width=True
        )

        st.markdown(
            "<div class='chart-caption'><b>Interpretation:</b> Category breakdown identifies which product lines drive "
            "the largest basket size (AOV) and revenue contribution.</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # Delivery Delay & Customer Satisfaction Impact
        st.subheader("3. Delivery Delay and Satisfaction Analysis")
        del_col1, del_col2 = st.columns(2)

        with del_col1:
            delay_summary = filtered_df[delay_col].value_counts().reset_index()
            delay_summary.columns = ["Delivery Status", "Order Count"]
            fig_delay = px.bar(
                delay_summary,
                x="Delivery Status",
                y="Order Count",
                color="Delivery Status",
                text="Order Count",
                title="On-Time vs Delayed Orders",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"}
            )
            fig_delay.update_traces(textposition="outside", texttemplate="%{text:,}")
            fig_delay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                showlegend=False,
                yaxis_title="Orders"
            )
            st.plotly_chart(fig_delay, use_container_width=True)

        with del_col2:
            rating_delay = (
                filtered_df.groupby(delay_col)[rating_col]
                .mean()
                .reset_index()
            )
            rating_delay.columns = ["Delivery Delay", "Average Rating"]
            fig_rating_impact = px.bar(
                rating_delay,
                x="Delivery Delay",
                y="Average Rating",
                color="Delivery Delay",
                text="Average Rating",
                title="Impact of Delivery Delay on Service Rating",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"}
            )
            fig_rating_impact.update_traces(textposition="outside", texttemplate="%{text:.2f} ⭐")
            fig_rating_impact.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                showlegend=False,
                yaxis_range=[0, 5.5],
                yaxis_title="Average Rating (1-5)"
            )
            st.plotly_chart(fig_rating_impact, use_container_width=True)

        st.markdown(
            "<div class='chart-caption'><b>Interpretation:</b> Delivery delays show a direct correlation with lower customer satisfaction ratings, "
            "highlighting the operational necessity of strict delivery SLAs in quick commerce.</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # Customer Feedback & Refund Request Analysis
        st.subheader("4. Refund Request Analysis")
        ref_col1, ref_col2 = st.columns(2)

        with ref_col1:
            refund_by_platform = (
                filtered_df.groupby([platform_col, refund_col])
                .size()
                .reset_index(name="Orders")
            )
            fig_ref_plat = px.bar(
                refund_by_platform,
                x=platform_col,
                y="Orders",
                color=refund_col,
                barmode="group",
                title="Refund Requests Across Platforms",
                color_discrete_map={"No": "#3B82F6", "Yes": "#F59E0B"}
            )
            fig_ref_plat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                yaxis_title="Orders",
                legend_title="Refund Requested"
            )
            st.plotly_chart(fig_ref_plat, use_container_width=True)

        with ref_col2:
            refund_delay_crosstab = (
                pd.crosstab(
                    filtered_df[delay_col],
                    filtered_df[refund_col],
                    normalize="index"
                ) * 100
            ).round(1).reset_index()
            
            fig_ref_delay = px.bar(
                refund_delay_crosstab,
                x=delay_col,
                y=["No", "Yes"],
                title="Refund Request Rate by Delay Status (%)",
                labels={"value": "Percentage (%)", "variable": "Refund Requested"},
                color_discrete_map={"No": "#94A3B8", "Yes": "#EF4444"}
            )
            fig_ref_delay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                barmode="stack",
                yaxis_title="Percentage (%)",
                xaxis_title="Delivery Delayed?"
            )
            st.plotly_chart(fig_ref_delay, use_container_width=True)

        st.markdown(
            "<div class='chart-caption'><b>Interpretation:</b> Orders experiencing delivery delays exhibit significantly higher refund request rates, "
            "directly increasing post-order operational costs.</div>",
            unsafe_allow_html=True
        )


# =============================================================================
# TAB 4: VISUALIZATIONS
# =============================================================================
with tab4:
    if filtered_df.empty:
        st.warning("No records match the active filter criteria. Please adjust your sidebar filters or click 'Reset Filters'.")
    else:
        st.subheader("Interactive Visual Analytics")

        # ---------------------------------------------------------------------
        # SECTION A: BAR CHARTS
        # ---------------------------------------------------------------------
        st.markdown("#### A. Comparative Bar Charts")
        b_col1, b_col2 = st.columns(2)

        with b_col1:
            # 1. Orders by Platform
            orders_platform = (
                filtered_df.groupby(platform_col)
                .size()
                .reset_index(name="Orders")
                .sort_values("Orders", ascending=False)
            )
            fig_bp1 = px.bar(
                orders_platform,
                x=platform_col,
                y="Orders",
                color=platform_col,
                text="Orders",
                title="Total Orders by Platform",
                color_discrete_sequence=["#2563EB", "#0D9488", "#F59E0B"]
            )
            fig_bp1.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_bp1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                showlegend=False,
                yaxis_title="Orders",
                xaxis_title="Platform"
            )
            st.plotly_chart(fig_bp1, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Compares order volume across quick-commerce providers.</div>",
                unsafe_allow_html=True
            )

        with b_col2:
            # 2. Revenue by Platform
            rev_platform = (
                filtered_df.groupby(platform_col)[value_col]
                .sum()
                .reset_index()
                .sort_values(value_col, ascending=False)
            )
            fig_bp2 = px.bar(
                rev_platform,
                x=platform_col,
                y=value_col,
                color=platform_col,
                text=value_col,
                title="Gross Revenue by Platform (INR)",
                color_discrete_sequence=["#1E40AF", "#0F766E", "#D97706"]
            )
            fig_bp2.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
            fig_bp2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                showlegend=False,
                yaxis_title="Revenue (INR)",
                xaxis_title="Platform"
            )
            st.plotly_chart(fig_bp2, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Highlights total commercial revenue captured by each platform.</div>",
                unsafe_allow_html=True
            )

        b_col3, b_col4 = st.columns(2)

        with b_col3:
            # 3. Average Delivery Time by Platform
            time_platform = (
                filtered_df.groupby(platform_col)[delivery_col]
                .mean()
                .reset_index()
                .sort_values(delivery_col)
            )
            fig_bp3 = px.bar(
                time_platform,
                x=platform_col,
                y=delivery_col,
                color=platform_col,
                text=delivery_col,
                title="Average Delivery Time Across Platforms",
                color_discrete_sequence=px.colors.qualitative.Dark2
            )
            fig_bp3.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
            fig_bp3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                showlegend=False,
                yaxis_title="Avg Time (Minutes)",
                xaxis_title="Platform"
            )
            st.plotly_chart(fig_bp3, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Evaluates fulfillment speed efficiency among competitors.</div>",
                unsafe_allow_html=True
            )

        with b_col4:
            # 4. Orders by Product Category
            cat_orders = (
                filtered_df.groupby(category_col)
                .size()
                .reset_index(name="Orders")
                .sort_values("Orders", ascending=False)
            )
            fig_bp4 = px.bar(
                cat_orders,
                x=category_col,
                y="Orders",
                color=category_col,
                text="Orders",
                title="Orders by Product Category",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bp4.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_bp4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                showlegend=False,
                yaxis_title="Orders",
                xaxis_title="Category"
            )
            st.plotly_chart(fig_bp4, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Identifies consumer demand across grocery and FMCG categories.</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ---------------------------------------------------------------------
        # SECTION B: LINE CHARTS (MEANINGFUL ORDERED DIMENSIONS)
        # ---------------------------------------------------------------------
        st.markdown("#### B. Trend Line Charts (Ordered Dimensions)")
        l_col1, l_col2 = st.columns(2)

        # Meaningful delivery time brackets
        bins = [0, 15, 25, 35, 45, 60, 120]
        bin_labels = ["<15 min", "15-25 min", "25-35 min", "35-45 min", "45-60 min", ">60 min"]
        filtered_df["Delivery_Bin"] = pd.cut(
            filtered_df[delivery_col],
            bins=bins,
            labels=bin_labels,
            include_lowest=True
        )

        with l_col1:
            # 1. Customer Rating variations across Delivery-Time Groups
            rating_by_bin = (
                filtered_df.groupby("Delivery_Bin", observed=False)[rating_col]
                .mean()
                .reset_index()
                .dropna()
            )
            fig_lp1 = px.line(
                rating_by_bin,
                x="Delivery_Bin",
                y=rating_col,
                markers=True,
                title="Customer Rating Trend Across Delivery-Time Groups",
                labels={"Delivery_Bin": "Delivery Time Bracket", rating_col: "Average Rating"}
            )
            fig_lp1.update_traces(
                line=dict(color="#3B82F6", width=3),
                marker=dict(size=9, color="#2563EB")
            )
            fig_lp1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                yaxis_range=[1, 5],
                yaxis_title="Average Service Rating (1-5)",
                xaxis_title="Delivery Time Bracket"
            )
            st.plotly_chart(fig_lp1, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Uses ordered delivery-time buckets to track how customer "
                "satisfaction fluctuates with longer waiting times.</div>",
                unsafe_allow_html=True
            )

        with l_col2:
            # 2. Average Order Value across Delivery Time Groups
            aov_by_bin = (
                filtered_df.groupby("Delivery_Bin", observed=False)[value_col]
                .mean()
                .reset_index()
                .dropna()
            )
            fig_lp2 = px.line(
                aov_by_bin,
                x="Delivery_Bin",
                y=value_col,
                markers=True,
                title="Average Order Value Across Delivery-Time Groups",
                labels={"Delivery_Bin": "Delivery Time Bracket", value_col: "Average Order Value (INR)"}
            )
            fig_lp2.update_traces(
                line=dict(color="#10B981", width=3),
                marker=dict(size=9, color="#059669")
            )
            fig_lp2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                yaxis_title="Average Order Value (INR)",
                xaxis_title="Delivery Time Bracket"
            )
            st.plotly_chart(fig_lp2, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Analyzes whether high-ticket baskets correlate with "
                "different fulfillment durations due to basket complexity.</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ---------------------------------------------------------------------
        # SECTION C: SCATTER PLOTS (OPTIMIZED PERFORMANCE)
        # ---------------------------------------------------------------------
        st.markdown("#### C. Scatter Plots (Multivariate Relationships)")
        s_col1, s_col2 = st.columns(2)

        # Responsive sampling for smooth client-side browser performance with 100k rows
        sample_size = min(3000, len(filtered_df))
        scatter_sample = (
            filtered_df.sample(n=sample_size, random_state=42)
            if len(filtered_df) > sample_size else filtered_df
        )

        with s_col1:
            fig_sc1 = px.scatter(
                scatter_sample,
                x=delivery_col,
                y=rating_col,
                color=platform_col,
                opacity=0.65,
                title=f"Delivery Time vs Customer Rating (Sample: {sample_size:,} orders)",
                labels={delivery_col: "Delivery Time (Minutes)", rating_col: "Service Rating"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_sc1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                yaxis_range=[0.5, 5.5],
                xaxis_title="Delivery Time (Minutes)",
                yaxis_title="Service Rating (1-5)"
            )
            st.plotly_chart(fig_sc1, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Visualizes individual order ratings against fulfillment times. "
                "Helps identify ratings clusters and outlier delays.</div>",
                unsafe_allow_html=True
            )

        with s_col2:
            fig_sc2 = px.scatter(
                scatter_sample,
                x=value_col,
                y=delivery_col,
                color=platform_col,
                size=rating_col,
                opacity=0.7,
                hover_data=[order_col, category_col, rating_col],
                title=f"Order Value vs Delivery Time (Sample: {sample_size:,} orders)",
                labels={value_col: "Order Value (INR)", delivery_col: "Delivery Time (Minutes)"},
                color_discrete_sequence=px.colors.qualitative.Dark2
            )
            fig_sc2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                xaxis_title="Order Value (INR)",
                yaxis_title="Delivery Time (Minutes)"
            )
            st.plotly_chart(fig_sc2, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Examines if larger basket amounts impact delivery latency, "
                "with bubble sizes scaled by customer rating.</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ---------------------------------------------------------------------
        # SECTION D: ADDITIONAL VISUALIZATIONS (HISTOGRAM, BOX PLOT, HEATMAP)
        # ---------------------------------------------------------------------
        st.markdown("#### D. Distribution, Dispersion & Correlation")
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            mean_del = filtered_df[delivery_col].mean()
            median_del = filtered_df[delivery_col].median()
            fig_hist = px.histogram(
                filtered_df,
                x=delivery_col,
                nbins=25,
                title="Delivery Time Frequency Distribution",
                color_discrete_sequence=["#3B82F6"]
            )
            fig_hist.add_vline(
                x=mean_del,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Mean: {mean_del:.1f}m",
                annotation_position="top right"
            )
            fig_hist.add_vline(
                x=median_del,
                line_dash="dot",
                line_color="#10B981",
                annotation_text=f"Median: {median_del:.1f}m",
                annotation_position="top left"
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                xaxis_title="Delivery Time (Minutes)",
                yaxis_title="Frequency (Orders)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Shows the central tendency and spread of delivery times. "
                "Vertical lines demarcate arithmetic mean and median.</div>",
                unsafe_allow_html=True
            )

        with d_col2:
            fig_box = px.box(
                filtered_df,
                x=platform_col,
                y=delivery_col,
                color=platform_col,
                title="Delivery Time Dispersion by Platform (Box Plot)",
                color_discrete_sequence=["#2563EB", "#0D9488", "#F59E0B"]
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                showlegend=False,
                xaxis_title="Platform",
                yaxis_title="Delivery Time (Minutes)"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'><b>Interpretation:</b> Displays the median, interquartile range (IQR), and outliers "
                "to evaluate delivery consistency across platforms.</div>",
                unsafe_allow_html=True
            )

        # Correlation Heatmap
        st.subheader("Statistical Correlation Matrix (Numerical Attributes)")
        corr_df = filtered_df[[delivery_col, value_col, rating_col]].copy()
        corr_df["Delay_Flag"] = (filtered_df[delay_col] == "Yes").astype(int)
        corr_df["Refund_Flag"] = (filtered_df[refund_col] == "Yes").astype(int)
        corr_matrix = corr_df.corr().round(3)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Heatmap (Delivery, Basket Size, Rating, Delays & Refunds)"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=40, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(
            "<div class='chart-caption'><b>Interpretation:</b> Pearson correlation coefficients evaluate linear relationships. "
            "Values near 0 denote statistical independence, whereas positive/negative values identify co-movement between operational metrics.</div>",
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# 6. APPLICATION FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: var(--text-color); opacity: 0.65; font-size: 0.85rem; padding: 10px 0;'>"
    "Quick-Commerce Delivery & Customer Experience Analytics Dashboard"
    "</div>",
    unsafe_allow_html=True
)
