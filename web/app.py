import sys
import os
import io
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from PIL import Image
# import zoneinfo

# Ensure the parent directory is in the path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.models import inference_engine
# from core.pdf_reports import generate_prediction_pdf
from core.constants import (
    GOODS_INFO, COUNTRY_OPTIONS, PORT_OPTIONS, CURRENCY_OPTIONS, INCOTERM_OPTIONS, ALL_FEATURES, INSUFFICIENT_DATA_CODES
)

# Load the app logo
app_logo = Image.open("./assets/price-app-logo.png")
# Set maximum and minimum years in Year selection
max_year = datetime.now().year
min_year = max_year - 3

def get_user_inputs():
    """Renders the input fields and returns a dictionary of values."""
    # with st.sidebar:
    #     st.header("⚙️ Report Settings")
    #     # Get all available timezones and sort them
    #     all_tz = sorted(list(zoneinfo.available_timezones()))
    #     
    #     # Determine default index (Asia/Dhaka if available)
    #     try:
    #         default_tz_index = all_tz.index("Asia/Dhaka")
    #     except ValueError:
    #         default_tz_index = 0
    #         
    #     selected_tz = st.selectbox("Your Timezone (for PDF)", all_tz, index=default_tz_index)
    #     st.session_state["selected_tz"] = selected_tz

    try:
        default_country_index = COUNTRY_OPTIONS.index('BANGLADESH')
    except ValueError:
        default_country_index = 0

    col1, col2 = st.columns(2)
    
    # Store UI inputs in a temporary dictionary to map to ALL_FEATURES later
    ui_inputs = {}
    
    with col1:
        available_codes = inference_engine.get_available_codes()
        goods_code = st.selectbox("Goods Code", available_codes, key="goods_code")
        ui_inputs["EXPORTER'S COUNTRY"] = st.selectbox("Exporter Country", options=COUNTRY_OPTIONS, index=default_country_index)
        ui_inputs["SHIPMENT FROM"] = st.selectbox("Shipment From Port/Country", options=COUNTRY_OPTIONS, index=default_country_index)
        ui_inputs["YEAR"] = st.number_input("Trading Year", min_value=min_year, step=1, max_value=max_year, key="trade_year")
        ui_inputs["CURRENCY"] = st.selectbox("Currency (e.g., USD, EUR)", options=CURRENCY_OPTIONS)
        ui_inputs["TRADE-TERM"] = st.selectbox("Incoterm (e.g., FOB, CPT)", options=INCOTERM_OPTIONS)
        ui_inputs["EXPORTER"] = st.text_input("Exporter")

    with col2:
        goods_description = GOODS_INFO.get(goods_code, "No description available.")
        st.text_input(label="Goods Description", value=goods_description, disabled=True)
        ui_inputs["COUNTRY_OF_ORIGIN"] = st.selectbox("Country of Origin", options=COUNTRY_OPTIONS, index=default_country_index)
        ui_inputs["SHIPMENT TO"] = st.selectbox("Shipment To Port/Country", options=PORT_OPTIONS)
        ui_inputs["QUANTITY"] = st.number_input("Quantity", min_value=0.0, step=0.1)
        ui_inputs["TENOR OF PAYMENT"] = st.number_input("Tenor of Payment", min_value=0, step=1)
        ui_inputs["FREIGHT CHARGES"] = st.number_input("Freight Charge", min_value=0.0, step=0.1)
        ui_inputs["IMPORTER"] = st.text_input("Importer")
    
    tolerance_pct = st.slider("Prediction Tolerance (%)", min_value=1, max_value=50, value=15)
    predict_button = st.button("🔍 Predict Unit Price", width='stretch', key="predict_button")
    
    # Dynamically build input_data using ALL_FEATURES
    input_data = {feature: ui_inputs[feature] for feature in ALL_FEATURES if feature in ui_inputs}
    
    return {
        "input_data": input_data,
        "goods_code": goods_code,
        "tolerance": tolerance_pct / 100.0,
        "currency": ui_inputs.get("CURRENCY", "USD"),
        "goods_description": goods_description,
        "predict_button": predict_button
    }

def run_prediction(input_data, goods_code, tolerance):
    """Handles API call logic with fallback to local inference."""
    api_url = os.environ.get('EXTERNAL_API_URL') or os.environ.get('API_URL') or "http://localhost:8080"
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
            
            request_url = f"{api_url}/predict"
            # If using API Gateway, the key might be required in the query string
            if "gateway.dev" in api_url:
                request_url = f"{request_url}?key={api_key}"

            response = requests.post(request_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json().get("result")
            elif response.status_code in [401, 403]:
                st.toast("API Authentication Failed (Gateway). Falling back to internal model.", icon="🚫")
            else:
                st.toast(f"API Error ({response.status_code}). Falling back to internal model.", icon="⚠️")
        except Exception as e:
            st.toast(f"API unreachable. Falling back to internal model.", icon="⚠️")
            print(f"API Error: {e}")
            
    # Fallback to internal inference
    if result is None:
        result = inference_engine.predict(input_data, goods_code, tolerance=tolerance, include_shap=True, shap_threshold_mb=400)
    
    return result

def render_insights(feature_importance):
    """Renders SHAP feature importance chart."""
    if not feature_importance:
        st.warning("No feature insights available for this prediction.")
        return

    st.subheader("🧐 Prediction Insights")
    st.write("This chart shows how the top features influenced the final predicted price.")

    # Sort features by absolute contribution
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_features]
    values = [x[1] for x in sorted_features]

    # Assign colors: Red for pushing price UP, Green for pushing price DOWN
    colors = ['#EF553B' if v > 0 else '#00CC96' for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.4f}" for v in values],
        textposition='auto',
    ))

    fig.update_layout(
        xaxis_title="Price Contribution",
        yaxis_title="Feature",
        yaxis={'autorange': "reversed"},
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, width='stretch')
    
    st.caption("🔴 Red bars indicate features that increased the predicted price. 🟢 Green bars indicates ones that decreased it.")

def render_results(result, currency, goods_code, goods_description, input_data):
    """Renders the prediction results or error messages."""
    if "error" in result:
        st.error(result["error"])
        return

    predicted_price = result["predicted_price"]
    lower_bound = result["lower_bound"]
    upper_bound = result["upper_bound"]
    feature_importance = result.get("feature_importance", {})

    st.success(f"**Predicted Unit Price:** {currency} {predicted_price:,.2f}")
    st.info(f"Expected Range: {currency} {lower_bound:,.2f} – {upper_bound:,.2f}")

    st.write("Use this range to detect under/over invoicing against declared unit price.")
    st.markdown("---")

    # Render Explainable AI Insights
    render_insights(feature_importance)

    st.markdown("---")
    
    # # Professional Report Export
    # # Capture the exact time localized to the user's selected timezone
    # tz_name = st.session_state.get("selected_tz", "Asia/Dhaka")
    # report_time = datetime.now(zoneinfo.ZoneInfo(tz_name))
    # 
    # pdf_bytes = generate_prediction_pdf(result, input_data, goods_code, goods_description, report_time=report_time)
    # st.download_button(
    #     label="📄 Download Official Audit Report (PDF)",
    #     data=pdf_bytes,
    #     file_name=f"Audit_Report_{goods_code}_{report_time.strftime('%Y%m%d%H%M%S')}.pdf",
    #     mime="application/pdf"
    # )
    # Warning message for goods with insufficient data
    if goods_code in INSUFFICIENT_DATA_CODES:
        st.markdown(":red[!!! IMPORTANT !!! The prediction may not be accurate:]")
        st.write(f"Goods '{goods_code}: {goods_description}' did not have sufficient data for training.")

def render_batch_processing():
    """Renders the batch processing tab."""
    st.header("📂 Batch Data Processing")
    st.write("Upload an Excel file containing invoice details to calculate predicted prices for all rows.")
    
    st.info("The Excel file must contain columns: **YEAR, QUANTITY, TENOR OF PAYMENT, FREIGHT CHARGES, EXPORTER, EXPORTER'S COUNTRY, IMPORTER, COUNTRY_OF_ORIGIN, CURRENCY, TRADE-TERM, SHIPMENT FROM, SHIPMENT TO, HSCODE, and TOLERANCE (optional)**.")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded {len(df)} rows.")
            
            batch_btn = st.button("🚀 Process Batch Predictions")
            
            if batch_btn:
                with st.spinner("Processing large-scale audit..."):
                    processed_df = inference_engine.predict_batch(df)
                
                st.success("Batch processing complete!")
                st.dataframe(processed_df.head(20), width='stretch')
                
                # Download link for result
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    processed_df.to_excel(writer, index=False)
                
                st.download_button(
                    label="📥 Download Processed Results (Excel)",
                    data=output.getvalue(),
                    file_name=f"Batch_Audit_Results_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Error processing batch: {e}")

def render_market_insights(baseline_data, goods_code):
    """Renders the Market Insights tab with trend visualizations."""
    st.header("📈 Market Trends & Model Elasticity")
    
    goods_description = GOODS_INFO.get(goods_code, "No description available.")
    st.info(f"Currently analyzing trends for: **{goods_code}: {goods_description}**")
    st.write("Explore how the predicted unit price varies based on historical patterns and trade volumes.")
    
    col_a, col_b = st.columns(2)
    
    # Use baseline data from Single Assessment
    baseline = baseline_data.copy()

    with col_a:
        st.subheader("📅 Price Trend (by Year)")
        years = list(range(min_year, max_year+1))
        prices = []
        for y in years:
            test_data = baseline.copy()
            test_data['YEAR'] = y
            res = inference_engine.predict(test_data, goods_code)
            prices.append(res['predicted_price'])
        
        # Use currency from baseline for label
        currency = baseline.get('CURRENCY', 'USD')
        fig_year = px.line(x=years, y=prices, markers=True, 
                          labels={'x': 'Year', 'y': f'Predicted Price ({currency})'},
                          title=f"Averaged Market Trend for {goods_code}")
        fig_year.update_layout(height=400)
        st.plotly_chart(fig_year, width='stretch')
        st.caption("This shows the model's perception of price inflation/deflation over time for this HS code.")

    with col_b:
        st.subheader("⚖️ Price vs. Quantity (Economy of Scale)")
        quantities = [10, 50, 100, 500, 1000, 5000, 10000]
        q_prices = []
        for q in quantities:
            test_data = baseline.copy()
            test_data['QUANTITY'] = q
            res = inference_engine.predict(test_data, goods_code)
            q_prices.append(res['predicted_price'])
            
        fig_q = px.line(x=quantities, y=q_prices, markers=True, log_x=True,
                       labels={'x': 'Quantity (Log Scale)', 'y': f'Predicted Price ({currency})'},
                       title=f"Quantity Sensitivity for {goods_code}")
        fig_q.update_layout(height=400)
        st.plotly_chart(fig_q, width='stretch')
        st.caption("How unit price typically shifts with shipment volume (Economy of Scale).")

def inject_custom_css():
    """Injects custom CSS for a premium look."""
    st.html(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 25px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            color: darkgrey;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #888888 !important;
            font-weight: bold !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #888888 !important;
            font-weight: bold !important;
        }
        .stTabs [aria-selected="true"] {
            color: #888888 !important;
            font-weight: bold !important;
        }
        .stButton > button:first-child {
            background-color: darkslategrey;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton > button:first-child:hover {
            background-color: teal;
            text-shadow: 0 1px 0 black;
        }
        .st-key-predict_button button {
            display: flex;
            margin: 0 auto;
            width: 50%;
            justify-content: center;
            align-items: center;
        }
        </style>
        """
    )

def main():
    st.set_page_config(page_title="Price Verification Tool 🏷️", page_icon=app_logo, layout="centered")
    inject_custom_css()
    st.logo(app_logo, size='large')
    st.title("💰 Unit Price Assessment") 
    
    #tab_single, tab_batch, tab_market = st.tabs(["🎯 Single Data Assessment", "📂 Batch Data Processing", "📈 Market Trends and Elasticity"])
    tab_single, tab_batch = st.tabs(["🎯 Single Data Assessment", "📂 Batch Data Processing"])

    with tab_single:
        st.markdown("Use this tool to assess potential under/over-invoicing based on historical import data.")
        # 1. Get User Inputs
        ui = get_user_inputs()

        # 2. Trigger Prediction
        if ui["predict_button"]:
            result = run_prediction(ui["input_data"], ui["goods_code"], ui["tolerance"])
            
            # 3. Render Results
            render_results(result, ui["currency"], ui["goods_code"], ui["goods_description"], ui["input_data"])

    with tab_batch:
        render_batch_processing()
        
    #with tab_market:
    #    render_market_insights(ui["input_data"], ui["goods_code"])

if __name__ == "__main__":
    main()
