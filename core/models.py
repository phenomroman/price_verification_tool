import os
import joblib
import numpy as np
import pandas as pd
from .constants import MODELS_DIR, ALL_FEATURES

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
            tolerance (float): Fallback tolerance if not provided in the DataFrame.
        Returns:
            pd.DataFrame: Original DataFrame + 'predicted_price', 'lower_bound', 'upper_bound'.
        """
        # Create an internal uppercase copy to handle case-insensitivity easily
        df_work = df.copy()
        df_work.columns = [c.upper() for c in df_work.columns]
        
        # Verify required features (ALL_FEATURES are already uppercase)
        missing = [f for f in ALL_FEATURES if f not in df_work.columns]
        if missing:
             raise ValueError(f"Missing columns: {missing}")

        # Normalize target code column name
        actual_code_col = goods_code_col.upper()
        if actual_code_col not in df_work.columns:
            raise ValueError(f"Missing goods code column: {goods_code_col}")

        # Initialize results columns
        df_work['predicted_price'] = np.nan
        df_work['lower_bound'] = np.nan
        df_work['upper_bound'] = np.nan

        # Determine row-level tolerances (case-insensitive)
        row_tolerances = df_work['TOLERANCE'].fillna(tolerance) if 'TOLERANCE' in df_work.columns else tolerance

        # Group by goods code for efficiency
        for code, group in df_work.groupby(actual_code_col):
            pipeline = self.pipelines.get(str(code))
            if not pipeline:
                continue

            # Predict (features match uppercase ALL_FEATURES exactly)
            preds = pipeline.predict(group[ALL_FEATURES])

            # Update working DF
            df_work.loc[group.index, 'predicted_price'] = preds
            
            # Use row-specific or fallback tolerance
            tol = row_tolerances.loc[group.index] if isinstance(row_tolerances, pd.Series) else row_tolerances
            df_work.loc[group.index, 'lower_bound'] = preds * (1 - tol)
            df_work.loc[group.index, 'upper_bound'] = preds * (1 + tol)

        # Concatenate results back to the original DataFrame to preserve formatting
        return pd.concat([df, df_work[['predicted_price', 'lower_bound', 'upper_bound']]], axis=1)

# Global instance to avoid reloading on every import (if desired/applicable)
inference_engine = ModelInference()
