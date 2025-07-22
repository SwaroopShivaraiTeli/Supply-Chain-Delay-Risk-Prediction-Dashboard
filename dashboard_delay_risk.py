import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load("delay_risk_model.pkl")
le_country = joblib.load("label_encoder_supplier_country.pkl")
le_target = joblib.load("label_encoder_target.pkl")

# Required features for the model
features = [
    'warehouse_inventory_level', 'handling_equipment_availability',
    'order_fulfillment_status', 'weather_condition_severity',
    'shipping_costs', 'supplier_reliability_score', 'lead_time_days',
    'historical_demand', 'cargo_condition_status', 'route_risk_level',
    'customs_clearance_time', 'disruption_likelihood_score',
    'supplier_country_encoded'
]

# Streamlit App UI
st.set_page_config(page_title="Supply Chain Delay Risk Predictor", layout="centered")
st.title("🚛 Supply Chain Delay Risk Predictor")
st.caption("📤 Upload your CSV file for batch prediction")

# Upload CSV
uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Encode 'supplier_country' to 'supplier_country_encoded'
        if "supplier_country" not in df.columns:
            st.error("❌ 'supplier_country' column is missing in your uploaded file.")
            st.stop()

        try:
            df["supplier_country_encoded"] = le_country.transform(df["supplier_country"])
        except Exception as e:
            st.error(f"⚠️ Error encoding supplier_country: {e}")
            st.stop()

        # Validate that all required features are present
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required feature columns in input data: {missing_cols}")
            st.stop()

        # Prediction
        X = df[features]
        predictions = model.predict(X)
        predicted_labels = le_target.inverse_transform(predictions)
        df["Predicted_Delay_Risk"] = predicted_labels

        # Show results
        st.subheader("📊 Predicted Delay Risk Levels")
        st.dataframe(df[["Predicted_Delay_Risk"]].value_counts().rename("Count").reset_index())

        # Plot risk distribution
        st.subheader("📈 Risk Distribution")
        fig, ax = plt.subplots()
        df["Predicted_Delay_Risk"].value_counts().plot(kind='bar', color='salmon', ax=ax)
        plt.xlabel("Delay Risk Level")
        plt.ylabel("Count")
        plt.title("Predicted Delay Risk Distribution")
        st.pyplot(fig)

        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predicted_delay_risk.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("📎 Please upload a CSV file to begin.")
