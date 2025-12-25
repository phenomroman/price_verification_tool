import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from .config import MODELS_DIR
from .constants import ALL_FEATURES

class ModelInference:
    def __init__(self):
        self.models = {}
        self.pipelines = {}
        self.cat_model = CatBoostRegressor()
        self.load_models()

    def load_models(self):
        """Loads models from the price_models directory."""
        if not os.path.exists(MODELS_DIR):
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return

        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".cbm"):
                code = filename.replace(".cbm", "")
                self.models[code] = self.cat_model.load_model(os.path.join(MODELS_DIR, filename))
            elif filename.endswith(".pkl"):
                code = filename.replace(".pkl", "")
                self.pipelines[code] = joblib.load(os.path.join(MODELS_DIR, filename))

    def get_available_codes(self):
        """Returns a list of available goods codes."""
        # Merge keys from both dictionaries
        codes = list(self.pipelines.keys()) + list(self.models.keys())
        return sorted(list(set(codes)))

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

        model = self.models.get(goods_code)
        pipeline = self.pipelines.get(goods_code)

        if not model and not pipeline:
            return {"error": f"No model found for code {goods_code}"}

        # Predict
        if pipeline:
            predicted_price = pipeline.predict(input_df)[0]
        else:
            predicted_price = model.predict(input_array.reshape(1, -1))[0]

        # Calculate range
        lower_bound = predicted_price * (1 - tolerance)
        upper_bound = predicted_price * (1 + tolerance)

        return {
            "predicted_price": predicted_price,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

# Global instance to avoid reloading on every import (if desired/applicable)
inference_engine = ModelInference()
