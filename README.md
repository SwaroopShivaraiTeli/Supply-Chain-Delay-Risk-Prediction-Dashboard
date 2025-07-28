# Supply-Chain-Delay-Risk-Prediction-Dashboard

📦 Supply Chain Delay Risk Predictor
A machine learning web app to predict delay risk levels in supply chain logistics using Random Forest, deployed via Streamlit. Upload your CSV data, get real-time predictions with delay risk classification (Low, Medium, High), and download results all with explainable AI features and robust validation.

🚀 Project Highlights
🔍 Delay Risk Classification using Random Forest (Accuracy: 98%)
🧠 Predicts risk level from 13 real-world logistics KPIs
🌍 Encodes supplier countries for regional analysis
📤 Upload CSV, get predictions, download results
📈 Visual summary of predicted risk levels
🧾 Custom label encoders saved and reused for production-ready deployment

🛠️ Tech Stack
Tool / Library	              Usage
pandas	                  Data handling and preprocessing
scikit-learn	            ML modeling (Random Forest)
joblib	                  Model serialization/deserialization
Streamlit	                Interactive web app UI
matplotlib	              Plotting prediction distributions

🔧 Setup Instructions (Local)
✅ Prerequisites:
Python 3.8+

pip (or conda)

📥 Install Dependencies:
bash
Copy
Edit
pip install -r requirements.txt

▶️ Run the App:
bash
Copy
Edit
streamlit run app.py
Then open http://localhost:8501 in your browser.

Ensure your input CSV has these required columns:
['warehouse_inventory_level', 'handling_equipment_availability',
 'order_fulfillment_status', 'weather_condition_severity',
 'shipping_costs', 'supplier_reliability_score', 'lead_time_days',
 'historical_demand', 'cargo_condition_status', 'route_risk_level',
 'customs_clearance_time', 'disruption_likelihood_score',
 'supplier_country']

📊 Model Performance (Test Set)
Metric	Score
Accuracy	98%
F1-Score	0.98
Precision	0.99
Recall	0.97

💡 Future Enhancements
✅ Add SHAP-based explainability for feature importance
🌐 Deploy to Streamlit Cloud or Hugging Face Spaces
🔒 Add data validation & file size limits
🧪 Build unit tests for input validation pipeline

Dashboard image is uploaded in the folder.

