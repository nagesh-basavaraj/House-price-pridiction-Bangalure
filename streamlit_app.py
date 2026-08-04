import streamlit as st
import pandas as pd
import joblib
import json
import os

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.main{
    background-color:#0f172a;
}

.stApp{
    background:#0f172a;
}

h1{
    color:#38bdf8;
    text-align:center;
}

h2,h3{
    color:white;
}

div[data-testid="stMetric"]{
    background:#1e293b;
    padding:18px;
    border-radius:15px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#38bdf8;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0ea5e9;
}

div[data-testid="stSidebar"]{
    background:#111827;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.image(
    "https://img.icons8.com/fluency/96/home.png",
    width=100
)

st.sidebar.title("🏠 House Price Prediction")

st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("---")

st.sidebar.write("### 👨‍💻 Developer")

st.sidebar.write("**B Nagesha**")

st.sidebar.write("Final Year B.Tech (AI & ML)")

st.sidebar.markdown("---")

st.sidebar.write("### ⚙ Technologies")

st.sidebar.write("✔ Python")

st.sidebar.write("✔ Streamlit")

st.sidebar.write("✔ Random Forest")

st.sidebar.write("✔ Pandas")

st.sidebar.write("✔ Scikit-Learn")

st.sidebar.write("✔ Joblib")

st.sidebar.markdown("---")

st.sidebar.success("Ready for Prediction 🚀")
# ======================================
# LOAD MODEL
# ======================================

MODEL_PATH = "house_price_model.pkl"

if os.path.exists(MODEL_PATH):

    model = joblib.load(MODEL_PATH)

else:

    st.error("❌ Model file (house_price_model.pkl) not found.")
    st.stop()

# ======================================
# LOAD FEATURE IMPORTANCE
# ======================================

feature_importance = {}

if os.path.exists("feature_importances.json"):

    with open("feature_importances.json", "r") as f:
        feature_importance = json.load(f)

# ======================================
# TITLE
# ======================================

st.title("🏠 Bangalore House Price Prediction")

st.markdown(
    """
Predict the estimated **house price** using a trained Machine Learning model.

Enter the property details below and click **Predict Price**.
"""
)

st.divider()

# ======================================
# INPUT SECTION
# ======================================

# ======================================
# PROPERTY DETAILS
# ======================================

st.subheader("🏡 Enter Property Details")

col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area (sq.ft)",
        min_value=300,
        max_value=10000,
        value=1500
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    floors = st.number_input(
        "Floors",
        min_value=1,
        max_value=5,
        value=2
    )

with col2:

    year = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2025,
        value=2005
    )

    location = st.selectbox(
        "Location",
        [
            "Downtown",
            "Suburban",
            "Urban",
            "Rural"
        ]
    )

    condition = st.selectbox(
        "Condition",
        [
            "Poor",
            "Fair",
            "Good",
            "Excellent"
        ]
    )

    garage = st.selectbox(
        "Garage",
        [
            "Yes",
            "No"
        ]
    )

st.divider()
# ======================================
# PREDICT HOUSE PRICE
# ======================================

if st.button("🏠 Predict House Price"):

    # Create DataFrame

    input_data = pd.DataFrame({

        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "YearBuilt": [year],
        "Location": [location],
        "Condition": [condition],
        "Garage": [garage]

    })

    try:

        prediction = model.predict(input_data)[0]

        st.success("✅ Prediction Successful")

        st.markdown("## 💰 Estimated House Price")

        st.markdown(f"""
        <div style="
            background:#16a34a;
            padding:25px;
            border-radius:15px;
            text-align:center;
            color:white;
            font-size:35px;
            font-weight:bold;">
            ₹ {prediction:,.2f}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        st.error("Prediction Failed!")

        st.exception(e)

        # ======================================
# PROPERTY SUMMARY
# ======================================

st.divider()

st.subheader("📋 Property Summary")

summary_df = pd.DataFrame({
    "Feature": [
        "Area (sq.ft)",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Year Built",
        "Location",
        "Condition",
        "Garage"
    ],
    "Value": [
        area,
        bedrooms,
        bathrooms,
        floors,
        year,
        location,
        condition,
        garage
    ]
})

st.dataframe(summary_df, use_container_width=True)

# ======================================
# FEATURE IMPORTANCE
# ======================================

if feature_importance:

    st.divider()

    st.subheader("📊 Feature Importance")

    feature_df = pd.DataFrame({
        "Feature": feature_importance.keys(),
        "Importance": feature_importance.values()
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        feature_df.set_index("Feature")
    )

    # ======================================
# DOWNLOAD PREDICTION REPORT
# ======================================

st.divider()

st.subheader("📥 Download Prediction Report")

report = pd.DataFrame({

    "Feature":[
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Year Built",
        "Location",
        "Condition",
        "Garage",
        "Predicted Price"
    ],

    "Value":[
        area,
        bedrooms,
        bathrooms,
        floors,
        year,
        location,
        condition,
        garage,
        prediction if 'prediction' in locals() else ""
    ]

})

csv = report.to_csv(index=False)

st.download_button(

    label="⬇ Download Report",

    data=csv,

    file_name="House_Price_Prediction_Report.csv",

    mime="text/csv"

)

# ======================================
# PROJECT METRICS
# ======================================

st.divider()

st.subheader("📈 Model Information")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with c2:
    st.metric(
        "Dataset",
        "House Price"
    )

with c3:
    st.metric(
        "Status",
        "Ready"
    )

# ======================================
# PROJECT DESCRIPTION
# ======================================

st.divider()

st.subheader("📖 About Project")

st.info("""

This project predicts Bangalore house prices using a Machine Learning model.

Technologies Used:

• Python

• Streamlit

• Pandas

• Scikit-Learn

• Joblib

Developed by **B Nagesha**

""")

# ======================================
# FOOTER
# ======================================

st.divider()

st.markdown(
"""
<center>

Made with ❤️ using Streamlit

© 2026 B Nagesha

</center>
""",
unsafe_allow_html=True
)