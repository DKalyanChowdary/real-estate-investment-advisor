import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

# ---------------------------------
# Google Drive Dataset Loader
# ---------------------------------
@st.cache_data
def load_data():
    file_id = "1mbGUFYsNGrqCT0UEyAM6kowpQ0Ixleu-"
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    df = pd.read_csv(download_url)

    df["price_per_sqft"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
    df["Future_Price_5yrs"] = df["Price_in_Lakhs"] * (1 + 0.08) ** 5
    df["Infrastructure_Score"] = df["Nearby_Schools"].fillna(0) + df["Nearby_Hospitals"].fillna(0)
    df["Amenities_Count"] = df["Amenities"].fillna("").apply(
        lambda x: len([a for a in str(x).split(",") if a.strip() != ""])
    )
    return df

# ---------------------------------
# Load trained models (AUTO DOWNLOAD FROM GOOGLE DRIVE)
# ---------------------------------

REG_ID = "1RKofV4OLuJP8xtGOu3yxkUq_i9tptgii"   # regression model
CLS_ID = "1EgcHqE9nTlyCXFAcY3coYLqGGd74rtnY"   # classification

REG_URL = f"https://drive.google.com/uc?id={REG_ID}&export=download"
CLS_URL = f"https://drive.google.com/uc?id={CLS_ID}&export=download"

REG_PATH = "regression_model.pkl"
CLS_PATH = "classification_model.pkl"

@st.cache_resource
def load_models():
    # Download regression model
    if not os.path.exists(REG_PATH):
        gdown.download(REG_URL, REG_PATH, quiet=False)

    # Download classification model
    if not os.path.exists(CLS_PATH):
        gdown.download(CLS_URL, CLS_PATH, quiet=False)

    reg = joblib.load(REG_PATH)
    cls = joblib.load(CLS_PATH)
    return reg, cls

reg_model, cls_model = load_models()
df = load_data()

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")
st.title("🏠 Real Estate Investment Advisor")
st.write("Predict future property value and check if it's a good investment.")

# -----------------------------------
# Sidebar Filters for Dataset View
# -----------------------------------
st.sidebar.header("Filter Properties")

min_size, max_size = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
size_filter = st.sidebar.slider("Size (SqFt)", min_size, max_size, (min_size, min_size + 300))

min_price, max_price = int(df["Price_in_Lakhs"].min()), int(df["Price_in_Lakhs"].max())
price_filter = st.sidebar.slider("Price (Lakhs)", min_price, max_price, (min_price, min_price + 30))

bhk_filter = st.sidebar.multiselect(
    "BHK Options", sorted(df["BHK"].dropna().unique()), default=[2, 3]
)

city_list = ["All"] + sorted(df["City"].dropna().unique())
city_filter = st.sidebar.selectbox("City", city_list)

mask = (
    df["Size_in_SqFt"].between(size_filter[0], size_filter[1])
    & df["Price_in_Lakhs"].between(price_filter[0], price_filter[1])
    & df["BHK"].isin(bhk_filter)
)

if city_filter != "All":
    mask &= df["City"] == city_filter

filtered_df = df[mask]

st.subheader("📄 Filtered Properties")
st.dataframe(filtered_df.head(50))

# -----------------------------------
# Visual Insights
# -----------------------------------
st.subheader("📊 City Visual Insights")

# City Price per SqFt
if "City" in df.columns:
    city_pps = df.groupby("City")["price_per_sqft"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=city_pps.index[:20], y=city_pps.values[:20], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Trend chart (Median Price per BHK)
bhk_trend = df.groupby("BHK")["Price_in_Lakhs"].median()
fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.lineplot(x=bhk_trend.index, y=bhk_trend.values, marker="o", ax=ax2)
ax2.set_title("Median Price by BHK")
st.pyplot(fig2)

# -----------------------------------
# Prediction Form
# -----------------------------------
st.subheader("🔮 Property Prediction Form")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", df["City"].dropna().unique())
    property_type = st.selectbox("Property Type", df["Property_Type"].dropna().unique())
    size = st.number_input("Size (SqFt)", 300, 10000, 1000)
    bhk = st.number_input("BHK", 1, 10, 2)
    age = st.number_input("Age of Property", 0, 50, 5)

with col2:
    schools = st.number_input("Nearby Schools", 0, 20, 3)
    hospitals = st.number_input("Nearby Hospitals", 0, 20, 1)
    total_floors = st.number_input("Total Floors", 1, 50, 5)
    amenities = st.number_input("Amenities Count", 0, 20, 5)
    availability = st.selectbox("Availability Status", df["Availability_Status"].dropna().unique())

infra_score = schools + hospitals

input_df = pd.DataFrame([{
    "Size_in_SqFt": size,
    "BHK": bhk,
    "Age_of_Property": age,
    "Nearby_Schools": schools,
    "Nearby_Hospitals": hospitals,
    "Total_Floors": total_floors,
    "Amenities_Count": amenities,
    "Infrastructure_Score": infra_score,
    "Property_Type": property_type,
    "Furnished_Status": "Unknown",
    "Public_Transport_Accessibility": "Unknown",
    "Parking_Space": "Unknown",
    "Security": "Unknown",
    "Facing": "Unknown",
    "Owner_Type": "Unknown",
    "Availability_Status": availability,
    "City": city
}])

if st.button("Predict"):
    future_price = reg_model.predict(input_df)[0]
    st.success(f"Estimated Price After 5 Years: ₹ {future_price:,.2f} Lakhs")

    class_pred = cls_model.predict(input_df)[0]
    prob = cls_model.predict_proba(input_df)[0][1]

    if class_pred == 1:
        st.success(f"Good Investment ✔ (Confidence: {prob:.2f})")
    else:
        st.error(f"Not a Good Investment ✖ (Confidence: {prob:.2f})")

# -----------------------------------
# Feature Importance
# -----------------------------------
st.subheader("📌 Feature Importance (Top 20)")

try:
    model = cls_model.named_steps["rf"]
    importances = model.feature_importances_
    pre = cls_model.named_steps["pre"]
    feature_names = pre.get_feature_names_out()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi, x="importance", y="feature", ax=ax3)
    ax3.set_title("Top Feature Importances")
    st.pyplot(fig3)

except Exception as e:
    st.write("Feature importance could not be displayed:", e)

