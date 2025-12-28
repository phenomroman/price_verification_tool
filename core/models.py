import os
import joblib
import numpy as np
import pandas as pd
from .config import MODELS_DIR
from .constants import ALL_FEATURES

class ModelInference:
    def __init__(self):
        self.pipelines = {}
        self.load_models()

    def load_models(self):
        """Loads models from the price_models directory."""
        if not os.path.exists(MODELS_DIR):
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return

        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".pkl"):
                code = filename.replace(".pkl", "")
                self.pipelines[code] = joblib.load(os.path.join(MODELS_DIR, filename))

    def get_available_codes(self):
        """Returns a list of available goods codes."""
        return sorted(list(self.pipelines.keys()))

    def predict(self, input_data: dict, goods_code: str, tolerance: float = 0.15) -> dict:
        """
        Predicts unit price based on input data.
        
        Args:
            input_data (dict): Dictionary containing input features.
            goods_code (str): The HS code of the goods.
            
        Returns:
            dict: Contains 'predicted_price', 'lower_bound', 'upper_bound'.
        """
        # Prepare input array/df
        # The order must match ALL_FEATURES from constants
        
        try:
            row = [input_data.get(feature) for feature in ALL_FEATURES]
        except KeyError as e:
             return {"error": f"Missing feature: {e}"}

        input_array = np.array([row], dtype=object)
        input_df = pd.DataFrame(data=input_array, columns=ALL_FEATURES)

        pipeline = self.pipelines.get(goods_code)

        if not pipeline:
            return {"error": f"No model found for code {goods_code}"}

        # Predict
        predicted_price = pipeline.predict(input_df)[0]

        # Calculate range
        lower_bound = predicted_price * (1 - tolerance)
        upper_bound = predicted_price * (1 + tolerance)

        return {
            "predicted_price": float(predicted_price),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }

    def predict_batch(self, df: pd.DataFrame, goods_code_col: str = 'HSCODE', tolerance: float = 0.15) -> pd.DataFrame:
        """
        Predicts unit prices for a batch of entries in a DataFrame.
        Args:
            df (pd.DataFrame): DataFrame containing input features and a goods code column.
            goods_code_col (str): The column name for HS codes.  
        Returns:
            pd.DataFrame: Original DataFrame + 'predicted_price', 'lower_bound', 'upper_bound'.
        """
        # Ensure all features exist
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        if missing:
             raise ValueError(f"Missing columns: {missing}")

        # Initialize results columns
        df = df.copy()
        df['predicted_price'] = np.nan
        df['lower_bound'] = np.nan
        df['upper_bound'] = np.nan

        # Group by goods code for efficiency
        for code, group in df.groupby(goods_code_col):
            code_str = str(code)
            pipeline = self.pipelines.get(code_str)

            if not pipeline:
                continue # Skip or log error in a real scenario

            # Prepare data for this group
            X = group[ALL_FEATURES]
            
            # Predict
            preds = pipeline.predict(X)

            # Update DF using original indices
            df.loc[group.index, 'predicted_price'] = preds
            df.loc[group.index, 'lower_bound'] = preds * (1 - tolerance)
            df.loc[group.index, 'upper_bound'] = preds * (1 + tolerance)

        return df

# Global instance to avoid reloading on every import (if desired/applicable)
inference_engine = ModelInference()
