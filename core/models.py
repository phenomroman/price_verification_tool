import os
import joblib
import numpy as np
import pandas as pd
import shap
from .constants import MODELS_DIR, ALL_FEATURES

class ModelInference:
    def __init__(self):
        self.pipelines = {}
        self.explainers = {}
        self.preprocessors = {}
        self.feature_names = {}
        self.load_models()

    def load_models(self):
        """Loads models and initializes SHAP explainers."""
        if not os.path.exists(MODELS_DIR):
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return

        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".pkl"):
                code = filename.replace(".pkl", "")
                try:
                    pipeline = joblib.load(os.path.join(MODELS_DIR, filename))
                    self.pipelines[code] = pipeline
                    
                    # Initialize TreeExplainer and extract steps
                    model = pipeline
                    preprocessor = None
                    if hasattr(pipeline, "named_steps"):
                        preprocessor = pipeline.named_steps.get('preprocess')
                        model = pipeline.named_steps.get('model', pipeline)
                        
                        # Fallback discovery if names don't match
                        if model == pipeline:
                            for step in reversed(list(pipeline.named_steps.values())):
                                if hasattr(step, "get_feature_importance") or "catboost" in str(type(step)).lower():
                                    model = step
                                    break
                    
                    self.explainers[code] = shap.TreeExplainer(model)
                    self.preprocessors[code] = preprocessor
                    
                    # Store feature names for this model
                    extracted_names = None
                    if preprocessor:
                        if hasattr(preprocessor, "get_feature_names_out"):
                            extracted_names = preprocessor.get_feature_names_out()
                        elif hasattr(preprocessor, "get_feature_names"):
                            extracted_names = preprocessor.get_feature_names()
                    
                    if extracted_names is None:
                        if hasattr(model, "feature_names_"):
                            extracted_names = model.feature_names_
                    
                    # Convert to list and clean up scikit-learn prefixes (e.g., 'num__QUANTITY' -> 'QUANTITY')
                    if extracted_names is not None:
                        try:
                            # Safely handle potential index-like outputs
                            if len(extracted_names) > 0 and isinstance(extracted_names[0], (int, np.integer)):
                                print(f"Warning: Discovered integer feature names for {code}, falling back to ALL_FEATURES.")
                                extracted_names = None
                            else:
                                cleaned_names = []
                                for n in extracted_names:
                                    n_str = str(n)
                                    cleaned_names.append(n_str.split("__", 1)[-1] if "__" in n_str else n_str)
                                
                                # If count matches ALL_FEATURES, use original constants for perfect casing/spacing
                                if len(cleaned_names) == len(ALL_FEATURES):
                                    extracted_names = ALL_FEATURES
                                else:
                                    extracted_names = cleaned_names
                        except Exception as e:
                            print(f"Name cleaning failed for {code}: {e}")
                            extracted_names = None
                    
                    if extracted_names is not None:
                        self.feature_names[code] = extracted_names
                    else:
                        self.feature_names[code] = ALL_FEATURES
                        
                    print(f"SHAP explainer initialized for {code}")
                except Exception as e:
                    print(f"Error loading model or initializing SHAP for {code}: {e}")

    def get_available_codes(self):
        """Returns a list of available goods codes."""
        return sorted(list(self.pipelines.keys()))

    def predict(self, input_data: dict, goods_code: str, tolerance: float = 0.15) -> dict:
        """
        Predicts unit price based on input data and calculates feature contributions (SHAP).
        Args:
            input_data (dict): Dictionary containing input features.
            goods_code (str): The HS code of the goods.   
        Returns:
            dict: Contains 'predicted_price', 'lower_bound', 'upper_bound'.
        """
        # The input array/df order must match ALL_FEATURES from constants
        try:
            row = [input_data.get(feature) for feature in ALL_FEATURES]
        except KeyError as e:
             return {"error": f"Missing feature: {e}"}

        input_array = np.array([row], dtype=object)
        input_df = pd.DataFrame(data=input_array, columns=ALL_FEATURES)

        pipeline = self.pipelines.get(goods_code)
        explainer = self.explainers.get(goods_code)
        preprocessor = self.preprocessors.get(goods_code)
        feature_names = self.feature_names.get(goods_code, ALL_FEATURES)

        if not pipeline:
            return {"error": f"No model found for code {goods_code}"}

        # Predict
        predicted_price = pipeline.predict(input_df)[0]

        # Calculate SHAP values for explainability
        feature_importance = {}
        if explainer:
            try:
                # If preprocessor exists, we must transform raw features to match model input
                data_for_shap = input_df
                if preprocessor:
                    # Note: transform may return a numpy array
                    transformed = preprocessor.transform(input_df)
                    if isinstance(transformed, np.ndarray):
                        data_for_shap = pd.DataFrame(transformed, columns=feature_names)
                    else:
                        data_for_shap = transformed

                # TreeExplainer.shap_values returns values
                shap_values = explainer.shap_values(data_for_shap)
                
                # Robust extraction of the first sample's SHAP values
                if isinstance(shap_values, list):
                    # List of arrays [class_0, class_1, ...]
                    values = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
                elif len(shap_values.shape) == 3:
                    # (samples, features, classes)
                    values = shap_values[0, :, 0]
                elif len(shap_values.shape) == 2:
                    # (samples, features)
                    values = shap_values[0]
                else:
                    values = shap_values
                
                # Select final names based on count match
                final_names = feature_names
                if len(values) != len(final_names):
                     # If model input count doesn't match our name list, use generic string names
                     final_names = [f"Feature {i}" for i in range(len(values))]

                # Map values to the appropriate feature names (ensure string keys)
                feature_importance = {
                    str(feature): float(val) 
                    for feature, val in zip(final_names, values)
                }
            except Exception as e:
                print(f"SHAP calculation error for {goods_code}: {e}")
                # Fallback: try default Feature Importance if SHAP fails
                try:
                    # Re-extract model step if needed
                    model = pipeline
                    if hasattr(pipeline, "named_steps"):
                        model = pipeline.named_steps.get('model', pipeline)
                        if model == pipeline:
                            for step in reversed(list(pipeline.named_steps.values())):
                                if hasattr(step, "get_feature_importance") or hasattr(step, "feature_importances_"):
                                    model = step
                                    break
                    
                    # Use get_feature_importance or feature_importances_
                    if hasattr(model, "get_feature_importance"):
                         fi = model.get_feature_importance()
                    elif hasattr(model, "feature_importances_"):
                        fi = model.feature_importances_
                    else:
                        raise ValueError("No importance attribute found")
                        
                    feature_importance = {f: float(i) for f, i in zip(feature_names, fi)}
                    print(f"Fallback to Importance for {goods_code}")
                except Exception as fe:
                    print(f"Fallback failed for {goods_code}: {fe}")

        # Calculate price range
        lower_bound = predicted_price * (1 - tolerance)
        upper_bound = predicted_price * (1 + tolerance)

        return {
            "predicted_price": float(predicted_price),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "feature_importance": feature_importance
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

# Global instance
inference_engine = ModelInference()
