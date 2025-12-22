import os, joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load models
models = {}
for filename in os.listdir("price_models"):
  if filename.endswith(".pkl"):
    code = filename.replace(".pkl", "")
    models[code] = joblib.load(f"price_models/{filename}")

# Input features
year_features = ['YEAR']
num_features = ['QUANTITY', 'TENOR OF PAYMENT', 'FREIGHT CHARGES']
cat_features = ["EXPORTER", "EXPORTER'S COUNTRY", "IMPORTER", "COUNTRY_OF_ORIGIN", 
                "CURRENCY", "TRADE-TERM", "SHIPMENT FROM", "SHIPMENT TO"]
all_features = year_features + num_features + cat_features

# Input-Output process function
def process_model(input=[], code='52094200', tolerance=0.15):
  input_array = np.array([input], dtype=object)
  input_df = pd.DataFrame(data=input_array, columns=all_features)
  output = models[code].predict(input_df)[0]
  lower_bound = output * (1 - tolerance)
  upper_bound = output * (1 + tolerance)
  return {
    'output': output, 'lower_bound': lower_bound, 'upper_bound': upper_bound
  }
