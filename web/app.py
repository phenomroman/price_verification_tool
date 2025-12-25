import sys
import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure the parent directory is in the path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.models import inference_engine
from core.constants import (
    GOODS_INFO, COUNTRY_OPTIONS, PORT_OPTIONS, CURRENCY_OPTIONS, INCOTERM_OPTIONS, ALL_FEATURES
)
try:
    default_country_index = COUNTRY_OPTIONS.index('BANGLADESH')
except ValueError:
    default_country_index = 0

st.set_page_config(page_title="Price Verification Tool 🏷️", layout="centered")

st.title("💰 Unit Price Assessment")
st.markdown("Use this tool to assess potential under/over-invoicing based on historical import data.")

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    available_codes = inference_engine.get_available_codes()
    goods_code = st.selectbox("Goods Code", available_codes)
    exporter_country = st.selectbox("Exporter Country", options=COUNTRY_OPTIONS, index=default_country_index)
    shipment_from = st.selectbox("Shipment From Port/Country", options=COUNTRY_OPTIONS, index=default_country_index)
    trade_year = st.number_input("Trading Year", min_value=2022, step=1, max_value=datetime.now().year)
    currency = st.selectbox("Currency (e.g., USD, EUR)", options=CURRENCY_OPTIONS)
    incoterm = st.selectbox("Incoterm (e.g., FOB, CPT)", options=INCOTERM_OPTIONS)
    exporter = st.text_input("Exporter")

with col2:
    goods_description = GOODS_INFO.get(goods_code, "No description available.")
    st.text_input(label="Goods Description", value=goods_description, disabled=True)
    origin_country = st.selectbox("Country of Origin", options=COUNTRY_OPTIONS, index=default_country_index)
    shipment_to = st.selectbox("Shipment To Port/Country", options=PORT_OPTIONS)
    quantity = st.number_input("Quantity", min_value=0.0, step=0.1)
    tenor = st.number_input("Tenor of Payment", min_value=0, step=1)
    freight = st.number_input("Freight Charge", min_value=0.0, step=0.1)
    importer = st.text_input("Importer")

# Construct input data dictionary mapping to ALL_FEATURES names
# Note: The mapping here must match exactly what was in app.py's input_array construction
# app.py: [trade_year, quantity, tenor, freight, exporter, exporter_country, importer, origin_country, currency, incoterm, shipment_from, shipment_to]
# constants.py ALL_FEATURES: 
# YEAR_FEATURES = ['YEAR'] -> trade_year
# NUM_FEATURES = ['QUANTITY', 'TENOR OF PAYMENT', 'FREIGHT CHARGES'] -> quantity, tenor, freight
# CAT_FEATURES = ["EXPORTER", "EXPORTER'S COUNTRY", "IMPORTER", "COUNTRY_OF_ORIGIN", "CURRENCY", "TRADE-TERM", "SHIPMENT FROM", "SHIPMENT TO"]
# -> exporter, exporter_country, importer, origin_country, currency, incoterm, shipment_from, shipment_to

input_data = {
    'YEAR': trade_year,
    'QUANTITY': quantity,
    'TENOR OF PAYMENT': tenor,
    'FREIGHT CHARGES': freight,
    'EXPORTER': exporter,
    "EXPORTER'S COUNTRY": exporter_country,
    'IMPORTER': importer,
    'COUNTRY_OF_ORIGIN': origin_country,
    'CURRENCY': currency,
    'TRADE-TERM': incoterm,
    'SHIPMENT FROM': shipment_from,
    'SHIPMENT TO': shipment_to
}

if st.button("🔍 Predict Unit Price"):
    api_url = os.environ.get('EXTERNAL_API_URL') or os.environ.get('API_URL') or "http://localhost:8000"
    result = None
    
    # Try API first if configured
    if api_url:
        try:
            # Convert dictionary input_data to list based on ALL_FEATURES order
            input_list = [input_data[feature] for feature in ALL_FEATURES]
            
            payload = {
                "input_list": input_list,
                "code": goods_code
            }
            
            response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
            if response.status_code == 200:
                try:
                    result = response.json().get("result")
                except Exception as e:
                    st.toast(f"API returned invalid JSON. Falling back to local model.", icon="⚠️")
                    print(f"JSON parse error: {e}")
            else:
                st.toast(f"API Error ({response.status_code}). Falling back to local model.", icon="⚠️")
        except requests.exceptions.RequestException as e:
            st.toast(f"API unreachable. Falling back to local model.", icon="⚠️")
            print(f"API Request Exception: {e}")
            
    # Fallback to local inference if result is still None
    if result is None:
        result = inference_engine.predict(input_data, goods_code)
    
    if "error" in result:
        st.error(result["error"])
    else:
        predicted_price = result["predicted_price"]
        lower_bound = result["lower_bound"]
        upper_bound = result["upper_bound"]

        st.success(f"**Predicted Unit Price:** {currency} {predicted_price:,.2f}")
        st.info(f"Expected Range: {currency} {lower_bound:,.2f} – {upper_bound:,.2f}")

        st.write("Use this range to detect under/over invoicing against declared unit price.")
        st.markdown("---")
        if goods_code == '58071000' or goods_code == '96061000':
            st.markdown(":red[!!! IMPORTANT !!! The prediction may not be accurate:]")
            st.write(f"Goods '{goods_code}: {goods_description}' did not have sufficient data for training.")

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        display: flex;
        margin: 0 auto;
        justify-content: center;
        align-items: center;
        background-color: teal;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1b5e20;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)
