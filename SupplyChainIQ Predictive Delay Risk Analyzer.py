#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load and Explore Dataset
import pandas as pd

df = pd.read_csv(r"D:\NEW DATA ANALYTICS\New folder\dynamic_supply_chain_logistics_dataset_with_country.csv")
print(df.shape)
print(df.info())


# In[2]:


#Create Target Variable
def delay_risk_label(prob):
    if prob <= 0.33:
        return "Low"
    elif prob <= 0.66:
        return "Medium"
    else:
        return "High"

df['delay_risk_level'] = df['delay_probability'].apply(delay_risk_label)


# In[3]:


#Encode Categorical Feature
from sklearn.preprocessing import LabelEncoder

# Encode supplier_country
le_country = LabelEncoder()
df['supplier_country_encoded'] = le_country.fit_transform(df['supplier_country'])

# Encode target class labels
le_target = LabelEncoder()
df['delay_risk_level_encoded'] = le_target.fit_transform(df['delay_risk_level'])


# In[4]:


#Train/Test Split
from sklearn.model_selection import train_test_split

features = [
    'warehouse_inventory_level', 'handling_equipment_availability',
    'order_fulfillment_status', 'weather_condition_severity',
    'shipping_costs', 'supplier_reliability_score', 'lead_time_days',
    'historical_demand', 'cargo_condition_status', 'route_risk_level',
    'customs_clearance_time', 'disruption_likelihood_score',
    'supplier_country_encoded'
]

X = df[features]
y = df['delay_risk_level_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# In[5]:


from sklearn.ensemble import RandomForestClassifier
import joblib

# Train
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Save the Random Forest model
joblib.dump(rf_model, "delay_risk_model.pkl")
print("✅ Model saved successfully.")


# In[6]:


# Evaluate Model
from sklearn.metrics import classification_report

y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[10]:


import joblib
joblib.dump(rf_model, "delay_risk_model.pkl")
joblib.dump(le_country, "label_encoder_supplier_country.pkl")
joblib.dump(le_target, "label_encoder_target.pkl")


# In[13]:


loaded_model = joblib.load("delay_risk_model.pkl")
print(type(loaded_model))


# In[ ]:




