import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
st.write(os.listdir())
st.write(os.listdir("datasets"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sales Analytics System", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Sales Analytics")
page = st.sidebar.radio("Navigation", ["Upload Data", "Dashboard", "Prediction", "Download Data"])

# ---------------- SESSION STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- UPLOAD PAGE ----------------
if page == "Upload Data":
    st.title("Upload Sales Dataset")

    file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # Standardize column names (important)
            df.columns = df.columns.str.strip()

            st.session_state.df = df

            st.success("File uploaded successfully")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error loading file: {e}")

# ---------------- DASHBOARD ----------------
elif page == "Dashboard":
    st.title("Sales Dashboard")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload dataset first")
        st.stop()

    # Clean data
    df = df.dropna()

    # Convert date
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')

    # ---------------- KPI SECTION ----------------
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)

    if "Sales" in df.columns:
        col1.metric("Total Sales", round(df["Sales"].sum(), 2))

    if "Quantity Ordered" in df.columns:
        col2.metric("Total Quantity", int(df["Quantity Ordered"].sum()))

    if "Price Each" in df.columns:
        col3.metric("Average Price", round(df["Price Each"].mean(), 2))

    st.markdown("---")

    # ---------------- FILTER ----------------
    if "City" in df.columns:
        city = st.selectbox("Select City", df["City"].unique())
        filtered = df[df["City"] == city]
    else:
        filtered = df

    # ---------------- CHARTS ----------------
    col1, col2 = st.columns(2)

    # Monthly Sales
    if "Month" in filtered.columns:
        monthly = filtered.groupby("Month")["Sales"].sum()
        col1.subheader("Monthly Sales")
        col1.line_chart(monthly)

    # Hourly Sales
    if "Hour" in filtered.columns:
        hourly = filtered.groupby("Hour")["Sales"].sum()
        col2.subheader("Sales by Hour")
        col2.bar_chart(hourly)

    st.markdown("---")

    # Top Products
    if "Product" in filtered.columns:
        st.subheader("Top Products")
        top_products = filtered.groupby("Product")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(top_products)

    # Sales by City
    if "City" in df.columns:
        st.subheader("Sales by City")
        city_sales = df.groupby("City")["Sales"].sum()
        st.bar_chart(city_sales)

    # Quantity vs Sales
    if "Quantity Ordered" in df.columns:
        st.subheader("Quantity vs Sales")
        st.scatter_chart(df[["Quantity Ordered", "Sales"]])

    # ---------------- DOWNLOAD FILTERED DATA ----------------
    st.markdown("---")
    st.subheader("Download Filtered Data")

    csv_filtered = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset",
        data=csv_filtered,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )

# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.title("Sales Prediction")

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")
        st.stop()

    df = df.dropna()

    # Check required columns
    if "Quantity Ordered" in df.columns and "Price Each" in df.columns and "Sales" in df.columns:

        X = df[["Quantity Ordered", "Price Each"]]
        y = df["Sales"]

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Values")

        qty = st.number_input("Quantity Ordered", min_value=1)
        price = st.number_input("Price Each", min_value=1.0)

        if st.button("Predict Sales"):
            prediction = model.predict([[qty, price]])
            st.success(f"Predicted Sales: {round(prediction[0], 2)}")

    else:
        st.error("Required columns missing for prediction")

# ---------------- DOWNLOAD PAGE ----------------
    st.divider()
st.subheader("Download Sample Datasets")

st.write("Download these datasets to test the application features.")

# CSV
with open("datasets/Sales Data.csv", "rb") as file:
    st.download_button(
        label="Download Sales Data CSV",
        data=file.read(),   # FIX HERE
        file_name="Sales_Data.csv",
        mime="text/csv"
    )
