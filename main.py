import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

class Payload(BaseModel):
    data: list

@app.post("/predict-inventory")
def predict_inventory(payload: Payload):
    try:
        if not payload.data:
            raise HTTPException(status_code=400, detail="No data received from n8n.")
        
        # Replaces local CSV file loads with incoming live data from MySQL via n8n
        df_ml = pd.DataFrame(payload.data)
        
        # Ensure required columns exist with safe defaults
        if 'Qty_Ordered' not in df_ml.columns:
            df_ml['Qty_Ordered'] = 100.0
        if 'Days_Delay' not in df_ml.columns:
            df_ml['Days_Delay'] = 0.0
            
        df_ml['Qty_Ordered'] = pd.to_numeric(df_ml['Qty_Ordered'], errors='coerce').fillna(100.0)
        df_ml['Days_Delay'] = pd.to_numeric(df_ml['Days_Delay'], errors='coerce').fillna(0.0)

        # 1. Dynamic Inventory & EOQ Calculations
        annual_demand = df_ml['Qty_Ordered'] * 365
        order_cost = 50.0
        holding_cost = 5.0
        
        df_ml['Optimal_EOQ'] = np.ceil(np.sqrt((2 * annual_demand * order_cost) / holding_cost))
        df_ml['Dynamic_Safety_Stock'] = np.ceil(1.96 * 2.5 * (annual_demand / 365))

        # 2. Criticality & Buffer Stockout Logic[cite: 1]
        df_ml['Buffer_Days_Available'] = df_ml['Dynamic_Safety_Stock'] / (annual_demand / 365)
        df_ml['Is_Stockout_Risk'] = (df_ml['Days_Delay'] > df_ml['Buffer_Days_Available']).astype(int)

        # 3. Machine Learning Model Training (Random Forest with Executive Weights)[cite: 1]
        feature_cols = [c for c in ['Supplier_ID', 'material', 'Qty_Ordered'] if c in df_ml.columns]
        X = df_ml[feature_cols]
        X_encoded = pd.get_dummies(X, drop_first=True)
        y = df_ml['Is_Stockout_Risk']

        if len(df_ml) > 5 and y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            # Executive weight configuration from your notebook[cite: 1]
            executive_weights = {0: 1, 1: 10}
            
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight=executive_weights
            )
            rf_model.fit(X_train, y_train)
            risk_probabilities = rf_model.predict_proba(X_encoded)[:, 1]
        else:
            risk_probabilities = np.where(df_ml['Days_Delay'] > 2, 0.75, 0.15)

        # 4. Final Risk Scoring & Status Indicator[cite: 1]
        df_ml['Predicted_Risk_Score'] = np.round(risk_probabilities * 100, 1)
        df_ml['Stockout_Risk_Status'] = np.where(
            df_ml['Predicted_Risk_Score'] >= 30.0, 
            'Critical Stockout', 
            'Safe Buffer'
        )

        # Return results to n8n for Power BI
        results = df_ml.to_dict(orient='records')
        return {
            "status": "success",
            "total_processed": len(results),
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
