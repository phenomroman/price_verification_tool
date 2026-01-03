import sys
import os
import requests
import streamlit as st
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

def get_user_inputs():
    """Renders the input fields and returns a dictionary of values."""
    try:
        default_country_index = COUNTRY_OPTIONS.index('BANGLADESH')
    except ValueError:
        default_country_index = 0

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
        predict_button = st.button("🔍 Predict Unit Price")

    with col2:
        goods_description = GOODS_INFO.get(goods_code, "No description available.")
        st.text_input(label="Goods Description", value=goods_description, disabled=True)
        origin_country = st.selectbox("Country of Origin", options=COUNTRY_OPTIONS, index=default_country_index)
        shipment_to = st.selectbox("Shipment To Port/Country", options=PORT_OPTIONS)
        quantity = st.number_input("Quantity", min_value=0.0, step=0.1)
        tenor = st.number_input("Tenor of Payment", min_value=0, step=1)
        freight = st.number_input("Freight Charge", min_value=0.0, step=0.1)
        importer = st.text_input("Importer")
        tolerance_pct = st.slider("Prediction Tolerance (%)", min_value=1, max_value=50, value=15)

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
    
    return {
        "input_data": input_data,
        "goods_code": goods_code,
        "tolerance": tolerance_pct / 100.0,
        "currency": currency,
        "goods_description": goods_description,
        "predict_button": predict_button
    }

def run_prediction(input_data, goods_code, tolerance):
    """Handles API call logic with fallback to local inference."""
    api_url = os.environ.get('EXTERNAL_API_URL') or os.environ.get('API_URL') or "http://localhost:8000"
    result = None
    
    # Try API first if configured
    if api_url:
        try:
            api_key = os.environ.get("API_KEY")
            if not api_key:
                st.toast("API_KEY environment variable is not set. Please configure it to use the API.")
            
            headers = {"X-API-KEY": api_key}
            payload = {
                "input_data": input_data,
                "code": goods_code,
                "tolerance": tolerance
            }
            
            response = requests.post(f"{api_url}/predict", json=payload, headers=headers, timeout=5)
            
            if response.status_code == 200:
                result = response.json().get("result")
            elif response.status_code == 403:
                st.toast("API Authentication Failed. Falling back to local model.", icon="🚫")
            else:
                st.toast(f"API Error ({response.status_code}). Falling back to local model.", icon="⚠️")
        except Exception as e:
            st.toast(f"API unreachable. Falling back to local model.", icon="⚠️")
            print(f"API Error: {e}")
            
    # Fallback to local inference
    if result is None:
        result = inference_engine.predict(input_data, goods_code, tolerance=tolerance)
    
    return result

def render_results(result, currency, goods_code, goods_description):
    """Renders the prediction results or error messages."""
    if "error" in result:
        st.error(result["error"])
        return

    predicted_price = result["predicted_price"]
    lower_bound = result["lower_bound"]
    upper_bound = result["upper_bound"]

    st.success(f"**Predicted Unit Price:** {currency} {predicted_price:,.2f}")
    st.info(f"Expected Range: {currency} {lower_bound:,.2f} – {upper_bound:,.2f}")

    st.write("Use this range to detect under/over invoicing against declared unit price.")
    st.markdown("---")
    
    if goods_code in ['58071000', '96061000']:
        st.markdown(":red[!!! IMPORTANT !!! The prediction may not be accurate:]")
        st.write(f"Goods '{goods_code}: {goods_description}' did not have sufficient data for training.")

def inject_custom_css():
    """Injects custom CSS for a premium look."""
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            display: flex;
            margin: 0 auto;
            margin-top: 15px;
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

def main():
    st.set_page_config(page_title="Price Verification Tool 🏷️", layout="centered")
    inject_custom_css()

    st.title("💰 Unit Price Assessment")
    st.markdown("Use this tool to assess potential under/over-invoicing based on historical import data.")

    # 1. Get User Inputs
    ui = get_user_inputs()

    # 2. Trigger Prediction
    if ui["predict_button"]:
        result = run_prediction(ui["input_data"], ui["goods_code"], ui["tolerance"])
        
        # 3. Render Results
        render_results(result, ui["currency"], ui["goods_code"], ui["goods_description"])

if __name__ == "__main__":
    main()
