import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("rfr_model.pkl")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("Model Information")

st.sidebar.markdown("""
**Model**

- Random Forest Regressor

**Input Features**

- Years at Company
- Satisfaction Level
- Average Monthly Hours

**Output**

- Estimated Salary
""")

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("💼 Employee Salary Prediction")
st.write(
    "Estimate employee salary using three workplace-related features."
)

st.divider()

# ---------------------------------------------------
# Inputs
# ---------------------------------------------------
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    years_at_company = st.slider(
        "Years at Company",
        min_value=0,
        max_value=20,
        value=3
    )

with col2:
    satisfaction_level = st.slider(
        "Satisfaction Level",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01
    )

with col3:
    avg_monthly_hours = st.slider(
        "Average Monthly Hours",
        min_value=120,
        max_value=310,
        value=200
    )

predict = st.button("Predict Salary", use_container_width=True)

st.divider()

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if predict:

    input_data = np.array([
        years_at_company,
        satisfaction_level,
        avg_monthly_hours
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success("Prediction generated successfully.")

    st.metric(
        label="Estimated Salary",
        value=f"₹ {prediction:,.0f}"
    )

    st.subheader("Input Summary")

    df = pd.DataFrame({
        "Feature": [
            "Years at Company",
            "Satisfaction Level",
            "Average Monthly Hours"
        ],
        "Value": [
            years_at_company,
            satisfaction_level,
            avg_monthly_hours
        ]
    })

    fig = px.bar(
        df,
        x="Feature",
        y="Value",
        text="Value",
        title="Selected Input Features"
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    if satisfaction_level < 0.30 and avg_monthly_hours > 280:
        st.warning(
            "Low satisfaction combined with high monthly working hours may indicate an unhealthy workload."
        )

    with st.expander("About this model"):
        st.markdown("""
- **Algorithm:** Random Forest Regressor
- **Inputs:** Years at Company, Satisfaction Level, Average Monthly Hours
- **Output:** Estimated Salary
        """)

else:
    st.info("Adjust the input values and click **Predict Salary**.")