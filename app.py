import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import holidays
from datetime import timedelta

# ------------------------
# PAGE CONFIG & THEME
# ------------------------
st.set_page_config(page_title="E‑Commerce Analytics", layout="wide", initial_sidebar_state="expanded")
# Optional streamlit theme tweaks can be set in ~/.streamlit/config.toml or via the app settings

# ------------------------
# HELPERS
# ------------------------
def safe_load(path="ecommerce_data.csv"):
    """Load CSV and perform light checks. Expect columns:
       CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice, ProductCategory, Country
    """
    df = pd.read_csv(path, parse_dates=['InvoiceDate'])
    # Ensure required columns exist (add defaults if missing)
    required = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
    # Optional columns
    if 'ProductCategory' not in df.columns:
        df['ProductCategory'] = 'Unknown'
    if 'Country' not in df.columns:
        df['Country'] = 'Unknown'
    return df

def preprocess(df):
    # Basic cleaning
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df.drop_duplicates()
    # Feature engineering
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Date'] = df['InvoiceDate'].dt.date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    return df

def compute_rfm(df, ref_date=None):
    if ref_date is None:
        ref_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency = ('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency = ('InvoiceNo', 'nunique'),
        Monetary = ('TotalPrice', 'sum')
    ).reset_index()
    # Score 1-5 using quantiles; Recency reversed (lower recency => better)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    def seg(row):
        if row['RFM_Score'] == '555': return 'Champion'
        if int(row['R_Score']) >= 4 and int(row['F_Score']) >= 4: return 'Loyal'
        if int(row['R_Score']) >= 4: return 'Recent'
        if int(row['F_Score']) >= 4: return 'Frequent'
        if int(row['M_Score']) >= 4: return 'Big Spender'
        return 'Other'
    rfm['Segment'] = rfm.apply(seg, axis=1)
    return rfm

def kpis_from(df):
    return {
        'Total Revenue': df['TotalPrice'].sum(),
        'Total Orders': df['InvoiceNo'].nunique(),
        'Active Customers': df['CustomerID'].nunique()
    }

def add_descrip(text):
    """Helper for consistent description styling"""
    st.markdown(f"<div style='color:#444;font-size:13px'>{text}</div>", unsafe_allow_html=True)

# ------------------------
# LOAD & PREPROCESS
# ------------------------
st.sidebar.title("Data & Filters")
data_path = st.sidebar.text_input("CSV path", value="ecommerce_data.csv")
try:
    raw_df = safe_load(data_path)
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

df = preprocess(raw_df)

# ------------------------
# INTERACTIVE FILTERS (SIDEBAR)
# ------------------------
min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

categories = ['All'] + sorted(df['ProductCategory'].dropna().unique().tolist())
countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())

selected_category = st.sidebar.selectbox("Product Category", categories, index=0)
selected_country = st.sidebar.selectbox("Country", countries, index=0)

# Precompute RFM for filter choices
rfm_all = compute_rfm(df)
segments = ['All'] + sorted(rfm_all['Segment'].unique().tolist())
selected_segment = st.sidebar.selectbox("Customer Segment", segments, index=0)

# Apply filters
mask = (df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] < end_date)
if selected_category != 'All':
    mask &= (df['ProductCategory'] == selected_category)
if selected_country != 'All':
    mask &= (df['Country'] == selected_country)
df_filtered = df[mask].copy()

# Apply segment filter by joining on CustomerID if requested
if selected_segment != 'All':
    rfm_filtered = rfm_all[rfm_all['Segment'] == selected_segment]
    df_filtered = df_filtered[df_filtered['CustomerID'].isin(rfm_filtered['CustomerID'])]

# ------------------------
# TOP KPI ROW
# ------------------------
st.title("🛒 E‑Commerce Analytics — Interactive Dashboard")
st.markdown("Use the sidebar to filter the data. Hover charts for details. Charts are interactive and exportable.")

kpis = kpis_from(df_filtered)
col1, col2, col3, col4 = st.columns([1.8,1.8,1.8,3])
col1.metric("Total Revenue", f"₹ {kpis['Total Revenue']:,.0f}")
col2.metric("Total Orders", f"{kpis['Total Orders']:,}")
col3.metric("Active Customers", f"{kpis['Active Customers']:,}")
col4.markdown("**Current Filters:**  \n• Date: {} to {}  \n• Category: {}  \n• Country: {}  \n• Segment: {}"
               .format(start_date.date(), (end_date - pd.Timedelta(days=1)).date(),
                       selected_category, selected_country, selected_segment))

st.markdown("---")

# ------------------------
# TIME SERIES & SEASONALITY
# ------------------------
with st.container():
    st.subheader("Sales Over Time 📈")
    c1, c2 = st.columns([3,1])
    # Prepare series
    daily = df_filtered.set_index('InvoiceDate').resample('D')['TotalPrice'].sum().reset_index()
    daily.rename(columns={'InvoiceDate':'Date','TotalPrice':'DailySales'}, inplace=True)
    if daily['DailySales'].isna().all():
        st.info("No sales in the selected filters/date range.")
    else:
        daily['MA7'] = daily['DailySales'].rolling(7, min_periods=1).mean()
        # Interactive line with tooltips and animation frame by month (if enough dates)
        fig_ts = px.line(daily, x='Date', y='DailySales', title='Daily Sales (with 7‑day MA)',
                         labels={'DailySales':'Daily Revenue', 'Date':'Date'}, hover_data={'MA7':':.2f'})
        fig_ts.add_traces(px.line(daily, x='Date', y='MA7', labels={'MA7':'7-day MA'}).data)
        fig_ts.update_traces(mode='lines', hovertemplate='%{x}<br>Revenue: ₹%{y:,.0f}')
        fig_ts.update_layout(transition={'duration':300})
        st.plotly_chart(fig_ts, use_container_width=True)
        # Description and KPIs under chart
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Avg Daily Revenue", f"₹ {daily['DailySales'].mean():,.0f}")
        col_b.metric("Peak Day", f"{daily.loc[daily['DailySales'].idxmax(),'Date'].date()}")
        col_c.metric("Max Daily Revenue", f"₹ {daily['DailySales'].max():,.0f}")
        add_descrip("Line chart shows daily revenue with a 7‑day moving average to smooth short-term volatility. Hover to see exact values.")

    # Seasonal decomposition (static with matplotlib)
    st.subheader("Seasonal Decomposition")
    if len(daily) >= 14:  # need enough points
        try:
            decomposition = seasonal_decompose(daily['DailySales'].fillna(method='ffill'), model='additive', period=7)
            fig = decomposition.plot()
            fig.set_size_inches(12,8)
            st.pyplot(fig)
            add_descrip("Decomposed series into trend, seasonal, and residual components (weekly seasonality assumed).")
        except Exception as e:
            st.warning("Decomposition error: " + str(e))
    else:
        st.info("Not enough data points for seasonal decomposition (need ~2+ weeks).")

st.markdown("---")

# ------------------------
# RFM SEGMENTATION SECTION
# ------------------------
with st.container():
    st.subheader("Customer Segmentation — RFM 🧍‍♀️")
    rfm = compute_rfm(df)
    # If segment filter in sidebar is used, show RFM for filtered customers as well
    rfm_display = rfm.copy()
    if selected_category != 'All' or selected_country != 'All' or (start_date and end_date):
        # compute RFM on filtered transactions; already done earlier if segment filter used
        rfm_display = compute_rfm(df_filtered)

    # Pie chart of segments with hover
    seg_counts = rfm_display['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment','Count']
    fig_seg = px.pie(seg_counts, names='Segment', values='Count', title='Customer Segment Distribution',
                     hole=0.35)
    fig_seg.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='%{label}: %{value} customers')
    st.plotly_chart(fig_seg, use_container_width=True)
    # Small table of top 5 customers by Monetary
    st.markdown("**Top 5 customers by spend**")
    top5 = rfm_display.sort_values('Monetary', ascending=False).head(5)[['CustomerID','Recency','Frequency','Monetary','Segment']]
    st.dataframe(top5.reset_index(drop=True), use_container_width=True)
    add_descrip("RFM segments are created using Recency (days since last purchase), Frequency (unique orders), and Monetary (total spend). Use segments for targeted marketing.")

st.markdown("---")

# ------------------------
# SALES FUNNEL
# ------------------------
with st.container():
    st.subheader("Acquisition Funnel & Conversion 🛒")
    # If real funnel stages not available, simulate using conversion rates or derive if columns exist
    # Here we allow user to provide basic funnel numbers via sliders for demonstration
    st.markdown("Adjust the funnel numbers to simulate dropoffs (or replace with real metrics if available).")
    st.write("Default values are illustrative; replace with real stage counts if available in your dataset.")
    f1, f2, f3, f4, f5 = st.columns(5)
    v_visit = f1.number_input("Visits", value=10000, min_value=0, step=100)
    v_view = f2.number_input("Product Views", value=7000, min_value=0, step=100)
    v_cart = f3.number_input("Add to Cart", value=4000, min_value=0, step=50)
    v_checkout = f4.number_input("Checkout", value=2000, min_value=0, step=50)
    v_purchase = f5.number_input("Purchases", value=1500, min_value=0, step=10)

    funnel_df = pd.DataFrame({
        "Stage":["Visits","Views","Add to Cart","Checkout","Purchase"],
        "Users":[v_visit, v_view, v_cart, v_checkout, v_purchase]
    })
    fig_funnel = px.funnel(funnel_df, x='Users', y='Stage', title='User Drop‑off Funnel', color='Stage')
    fig_funnel.update_layout(transition={'duration':300})
    st.plotly_chart(fig_funnel, use_container_width=True)
    # Insight card
    conv_rate = (v_purchase / v_visit * 100) if v_visit>0 else 0
    st.metric("Overall Conversion Rate", f"{conv_rate:.2f}%")
    add_descrip("Funnel visualizes drop-off from visits to purchases. Use it to identify where to focus UX or incentive changes.")

st.markdown("---")

# ------------------------
# GEO VISUALIZATION
# ------------------------
with st.container():
    st.subheader("Geographic Sales Map 🗺️")
    if 'Country' in df_filtered.columns:
        geo = df_filtered.groupby('Country', as_index=False)['TotalPrice'].sum().sort_values('TotalPrice', ascending=False)
        if geo.shape[0] == 0:
            st.info("No geo data for the current filters.")
        else:
            fig_map = px.choropleth(geo, locations='Country', locationmode='country names',
                                    color='TotalPrice', hover_name='Country',
                                    color_continuous_scale='Blues', title='Sales by Country')
            st.plotly_chart(fig_map, use_container_width=True)
            top_country = geo.iloc[0]
            st.metric("Top Country", f"{top_country['Country']} — ₹{top_country['TotalPrice']:,.0f}")
            add_descrip("Choropleth uses country names to show aggregate revenue by country. Use country filters for regional analysis.")
    else:
        st.info("Country column not found in data.")

st.markdown("---")

# ------------------------
# FESTIVAL IMPACT
# ------------------------
with st.container():
    st.subheader("Festival Impact 🎉")
    # Use holidays library for India by default; allow selection
    country_option = st.selectbox("Holiday Country (for festival detection)", options=['India', 'US', 'UnitedKingdom'], index=0)
    years = list({d.year for d in df['InvoiceDate']})
    if country_option == 'India':
        hols = holidays.India(years=years)
    elif country_option == 'US':
        hols = holidays.US(years=years)
    else:
        hols = holidays.UnitedKingdom(years=years)

    df_filtered['IsFestival'] = df_filtered['InvoiceDate'].dt.date.isin(hols)
    festival_summary = df_filtered.groupby('IsFestival')['TotalPrice'].sum().reset_index()
    festival_summary['Label'] = festival_summary['IsFestival'].map({True:'Festival Days', False:'Non-Festival Days'})
    fig_fest = px.bar(festival_summary, x='Label', y='TotalPrice', text='TotalPrice',
                      title='Festival vs Non-Festival Revenue', color='Label', color_discrete_map={'Festival Days':'#EF553B','Non-Festival Days':'#636EFA'})
    fig_fest.update_traces(texttemplate='₹%{y:,.0f}', textposition='outside')
    st.plotly_chart(fig_fest, use_container_width=True)
    add_descrip("Compares revenue on holiday dates vs non-holiday dates. Useful to plan inventory and marketing spend during festivals.")

st.markdown("---")

# ------------------------
# FOOTER / DOWNLOADS
# ------------------------
with st.container():
    st.markdown("### Export / Notes")
    st.write("You can download filtered data for further analysis.")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered CSV", csv, file_name="filtered_ecommerce.csv", mime='text/csv')
    st.markdown("Built with Streamlit • Use the sidebar to adjust filters and explore the dataset interactively.")