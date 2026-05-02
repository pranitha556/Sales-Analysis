import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sales Analytics System", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Sales Analytics System")
page = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Dashboard", "Prediction", "Download Data", "Sample Datasets"]
)

# ---------------- SESSION STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = None


# ================== UPLOAD PAGE ==================
if page == "Upload Data":
    st.title("Upload Dataset")

    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            df.columns = df.columns.str.strip()
            st.session_state.df = df

            st.success("File uploaded successfully")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error: {e}")


# ================== DASHBOARD ==================
elif page == "Dashboard":
    st.title("Sales Dashboard")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload dataset first")
        st.stop()

    df = df.dropna()

    # Convert date
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')

    # KPIs
    col1, col2, col3 = st.columns(3)

    if "Sales" in df.columns:
        col1.metric("Total Sales", round(df["Sales"].sum(), 2))

    if "Quantity Ordered" in df.columns:
        col2.metric("Total Quantity", int(df["Quantity Ordered"].sum()))

    if "Price Each" in df.columns:
        col3.metric("Average Price", round(df["Price Each"].mean(), 2))

    st.markdown("---")

    # Filter
    if "City" in df.columns:
        city = st.selectbox("Select City", df["City"].unique())
        filtered = df[df["City"] == city]
    else:
        filtered = df

    col1, col2 = st.columns(2)

    # Monthly
    if "Month" in filtered.columns:
        monthly = filtered.groupby("Month")["Sales"].sum()
        col1.subheader("Monthly Sales")
        col1.line_chart(monthly)

    # Hourly
    if "Hour" in filtered.columns:
        hourly = filtered.groupby("Hour")["Sales"].sum()
        col2.subheader("Sales by Hour")
        col2.bar_chart(hourly)

    st.markdown("---")

    # Top Products
    if "Product" in filtered.columns:
        st.subheader("Top Products")
        top = filtered.groupby("Product")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(top)

    # Sales by City
    if "City" in df.columns:
        st.subheader("Sales by City")
        city_sales = df.groupby("City")["Sales"].sum()
        st.bar_chart(city_sales)

    # Scatter
    if "Quantity Ordered" in df.columns:
        st.subheader("Quantity vs Sales")
        st.scatter_chart(df[["Quantity Ordered", "Sales"]])

    # Download filtered
    st.markdown("---")
    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )


# ================== PREDICTION ==================
elif page == "Prediction":
    st.title("Sales Prediction")

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")
        st.stop()

    df = df.dropna()

    if all(col in df.columns for col in ["Quantity Ordered", "Price Each", "Sales"]):

        X = df[["Quantity Ordered", "Price Each"]]
        y = df["Sales"]

        model = LinearRegression()
        model.fit(X, y)

        qty = st.number_input("Quantity Ordered", min_value=1)
        price = st.number_input("Price Each", min_value=1.0)

        if st.button("Predict"):
            result = model.predict([[qty, price]])
            st.success(f"Predicted Sales: {round(result[0], 2)}")

    else:
        st.error("Dataset must contain Quantity Ordered, Price Each, Sales")


# ================== DOWNLOAD DATA ==================
elif page == "Download Data":
    st.title("Download Dataset")

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")
        st.stop()

    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Full Dataset",
        data=csv,
        file_name="sales_data.csv",
        mime="text/csv"
    )


# ================== SAMPLE DATASETS ==================
elif page == "Sample Datasets":
    st.title("Sample Datasets")

    st.write("Download these datasets to test the application")

    # Dataset 1
    st.subheader("Basic Sales Dataset")

    sample1 = pd.DataFrame({
        "Order ID": range(1, 51),
        "Product": np.random.choice(["Phone", "Laptop", "Headphones"], 50),
        "Quantity Ordered": np.random.randint(1, 5, 50),
        "Price Each": np.random.randint(100, 1000, 50),
        "Order Date": pd.date_range(start="2026-01-01", periods=50),
        "Month": np.random.randint(1, 12, 50),
        "Sales": np.random.randint(100, 5000, 50),
        "City": np.random.choice(["Bangalore", "Delhi", "Mumbai"], 50),
        "Hour": np.random.randint(0, 24, 50)
    })

    csv1 = sample1.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Basic Dataset",
        data=csv1,
        file_name="sample_basic.csv",
        mime="text/csv"
    )

    st.dataframe(sample1.head())

    # Dataset 2
    st.subheader("Extended Sales Dataset")

    sample2 = pd.DataFrame({
        "Order ID": range(1, 101),
        "Product": np.random.choice(["Tablet", "Monitor", "Keyboard"], 100),
        "Quantity Ordered": np.random.randint(1, 10, 100),
        "Price Each": np.random.randint(200, 2000, 100),
        "Order Date": pd.date_range(start="2026-02-01", periods=100),
        "Month": np.random.randint(1, 12, 100),
        "Sales": np.random.randint(500, 10000, 100),
        "City": np.random.choice(["Chennai", "Hyderabad", "Pune"], 100),
        "Hour": np.random.randint(0, 24, 100)
    })

    csv2 = sample2.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Extended Dataset",
        data=csv2,
        file_name="sample_extended.csv",
        mime="text/csv"
    )

    st.dataframe(sample2.head())
