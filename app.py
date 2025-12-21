import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Load all models dynamically ---
models = {}
pipelines = {}
cat_model = CatBoostRegressor()
for filename in os.listdir("price_models"):
    if filename.endswith(".cbm"):
        code = filename.replace(".cbm", "")
        models[code] = cat_model.load_model(os.path.join("price_models", filename))
    elif filename.endswith(".pkl"):
        code = filename.replace(".pkl", "")
        pipelines[code] = joblib.load(os.path.join("price_models", filename))

# --- Input Variables ---
country_list = ['AFGHANISTAN', 'ALGERIA', 'ARGENTINA', 'AUSTRALIA', 'AUSTRIA', 'BANGLADESH', 'BELARUS', 'BELGIUM', 'BENIN', 'BHUTAN', 'BOSNIA AND HERZEGOVINA', 'BRAZIL', 'BULGARIA', 'BURKINA FASO', 'CAMBODIA', 'CAMEROON', 'CANADA', 'CHAD', 'CHILE', 'CHINA, PEOPLE’S REPUBLIC OF', 'CONGO', 'CROATIA', 'CZECH REPUBLIC', 'CZECH REPUBLIC ', 'DENMARK', 'ECUADOR', 'EGYPT', 'EL-SALVADOR', 'ESTONIA', 'EUROPE', 'FIJI', 'FINLAND', 'FRANCE', 'GERMANY', 'GREECE', 'GUYANA', 'HONGKONG', 'HUNGARY', 'ICELAND', 'INDIA', 'INDONESIA', 'IRAQ', 'IRELAND', 'ITALY', 'JAPAN', 'KIRIBATI, REPUBLIC OF', 'KOREA, REPUBLIC OF (SOUTH)', 'KUWAIT', 'KYRGYZSTAN', 'LATVIA', 'LEBANON', 'LIBERIA', 'LITHUANIA', 'LUXEMBOURG', 'MACAO', 'MACEDONIA', 'MADAGASCAR', 'MALAYSIA', 'MALI', 'MALTA', 'MAURITIUS', 'MEXICO', 'MOROCCO', 'MYANMAR', 'NAURU', 'NEPAL', 'NETHERLANDS', 'NEWZEALAND', 'NORWAY', 'OMAN', 'PAKISTAN', 'PANAMA', 'PAPUA NEW GUINEA', 'PARAGUAY', 'PERU', 'PHILIPPINES', 'POLAND', 'PORTUGAL', 'QATAR', 'ROMANIA', 'RUSSIAN FEDERATION', 'SAMOA', 'SAN MARINO', 'SAUDI ARABIA', 'SENEGAL', 'SERBIA', 'SIERRA LEONE', 'SINGAPORE', 'SLOVAK REPUBLIC', 'SLOVENIA', 'SOUDIA ARABIA', 'SOUTH AFRICA', 'SOUTH SUDAN', 'SPAIN', 'SRI-LANKA', 'SUADIA ARABIA', 'SWEDEN', 'SWITZERLAND', 'TAIWAN', 'TAJIKISTAN', 'TANZANIA', 'THAILAND', 'TOGO', 'TUNISIA', 'TURKEY', 'U.A.E.', 'U.K', 'U.S.A', 'UKRAINE', 'URUGUAY', 'VANUATU', 'VIETNAM', 'ZIMBABWE']
default_country = 'BANGLADESH'
try:
    default_country_index = country_list.index(default_country)
except ValueError:
    default_country_index = 0
today = datetime.now()
goods_info = {'52094200': 'FABRICS - WOVEN DENIM', 
              '58071000': 'ACCESSORIES - BADGE LABEL', 
              '48211000': 'ACCESSORIES - WOVEN LABEL', 
              '96071900': 'ACCESSORIES - ZIPPER', 
              '96071100': 'ACCESSORIES - ZIPPER', 
              '96061000': 'ACCESSORIES - SNAP BUTTON', 
              '59032010': 'FABRICS - POLY TEXTILE', 
              '39262090': 'ACCESSORIES - HANGER', 
              '62171000': 'ACCESSORIES - SCRAP FABRIC', 
              '60063200': 'FABRICS - SYNTHETIC',
              '96062200': 'ACCESSORIES - METAL BUTTON',
              '60041000': 'FABRICS - KNITTED'}

# Initiate webpage
st.set_page_config(page_title="Price Verification Tool 🏷️", layout="centered")
st.title("💰 Unit Price Assessment")
st.markdown("Use this tool to assess potential under/over-invoicing based on historical import data.")

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    goods_code = st.selectbox("Goods Code", list(pipelines.keys() if pipelines else models.keys()))
    exporter_country = st.selectbox("Exporter Country", options=country_list, index=default_country_index)
    shipment_from = st.selectbox("Shipment From Port/Country", options=country_list, index=default_country_index)
    trade_year = st.number_input("Trading Year", min_value=2022, step=1, max_value=today.year-1 if today.month==1 else today.year)
    currency = st.selectbox("Currency (e.g., USD, EUR)",  
                            options=['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'CNY', 'INR'])
    incoterm = st.selectbox("Incoterm (e.g., FOB, CPT)", 
                            options=['EXW', 'FCA', 'CPT', 'CIP', 'DAP (DAF, DES, DDU)', 'DPU', 'DDP', 'FAS', 'CIF', 'FOB', 'CNF (CFR)'])
    exporter = st.text_input("Exporter")
with col2:
    goods_description = goods_info.get(goods_code, "No description available.")
    st.text_input(label="Goods Description", value=goods_description, disabled=True)
    origin_country = st.selectbox("Country of Origin", options=country_list, index=default_country_index)
    shipment_to = st.selectbox("Shipment To Port/Country", 
                               options=['EPZ', 'CHITTAGONG', 'BENAPOL', 'Burimari', 'Others', 'SHAHAJALAL INT. AIRPORT'])
    quantity = st.number_input("Quantity", min_value=0.0, step=0.1)
    tenor = st.number_input("Tenor of Payment", min_value=0, step=1)
    freight = st.number_input("Freight Charge", min_value=0.0, step=0.1)
    importer = st.text_input("Importer")

# Construct feature array – set according to model’s training schema
input_array = np.array([[trade_year, quantity, tenor, freight, exporter, exporter_country, 
                importer, origin_country, currency, incoterm, shipment_from, shipment_to]],
                dtype=object)
year_features = ['YEAR']
num_features = ['QUANTITY', 'TENOR OF PAYMENT', 'FREIGHT CHARGES']
cat_features = ["EXPORTER", "EXPORTER'S COUNTRY", "IMPORTER", "COUNTRY_OF_ORIGIN", 
                "CURRENCY", "TRADE-TERM", "SHIPMENT FROM", "SHIPMENT TO"]
all_features = year_features + num_features + cat_features
input_df = pd.DataFrame(data=input_array, columns=all_features)

# Output on button press
if st.button("🔍 Predict Unit Price"):
    # Model without data preprocessing
    model = models.get(goods_code)
    # Pipeline with data preprocessing
    pipeline = pipelines.get(goods_code)
           
    # Predict price using model/pipeline
    predicted_price = pipeline.predict(input_df)[0] if pipeline else model.predict(input_array.reshape(1,-1))[0]
        
    # Define a prediction range (±15% as example)
    lower_bound = predicted_price * 0.85
    upper_bound = predicted_price * 1.15

    # Show output
    st.success(f"**Predicted Unit Price:** {currency} {predicted_price:,.2f}")
    st.info(f"Expected Range: {currency} {lower_bound:,.2f} – {upper_bound:,.2f}")
    st.write("Use this range to detect under/over invoicing against declared unit price.")
    st.markdown("---")
    if goods_code == '58071000' or goods_code == '96061000':
        st.markdown(":red[!!! IMPORTANT !!! The prediction may not be accurate:]")
        st.write(f"Goods '{goods_code}: {goods_description}' did not have sufficient data for training.")

# Styling
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
