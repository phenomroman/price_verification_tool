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
        self.available_codes = self._discover_models()

    def _discover_models(self):
        """Discovers available goods codes from the models directory."""
        if not os.path.exists(MODELS_DIR):
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return []
        
        codes = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
        print(f"Discovered {len(codes)} available models.")
        return sorted(codes)

    def _load_single_model(self, code, include_shap=True):
        """Loads a specific model and optionally initializes its SHAP explainer on demand."""
        if code in self.pipelines:
            # If already loaded but SHAP is now requested and was previously skipped
            if include_shap and code not in self.explainers:
                return self._init_shap(code)
            return True

        if code not in self.available_codes:
            return False

        file_path = os.path.join(MODELS_DIR, f"{code}.pkl")
        try:
            print(f"Loading model for {code}...")
            pipeline = joblib.load(file_path)
            self.pipelines[code] = pipeline
            
            # Extract model and preprocessor
            model = pipeline
            preprocessor = None
            if hasattr(pipeline, "named_steps"):
                preprocessor = pipeline.named_steps.get('preprocess')
                model = pipeline.named_steps.get('model', pipeline)
                if model == pipeline:
                    for step in reversed(list(pipeline.named_steps.values())):
                        if hasattr(step, "get_feature_importance") or "catboost" in str(type(step)).lower():
                            model = step
                            break
            
            self.preprocessors[code] = preprocessor

            # Extract feature names
            extracted_names = None
            if preprocessor:
                if hasattr(preprocessor, "get_feature_names_out"):
                    extracted_names = preprocessor.get_feature_names_out()
                elif hasattr(preprocessor, "get_feature_names"):
                    extracted_names = preprocessor.get_feature_names()
            
            if extracted_names is None and hasattr(model, "feature_names_"):
                extracted_names = model.feature_names_

            if extracted_names is not None:
                cleaned_names = []
                for n in extracted_names:
                    n_str = str(n)
                    cleaned_names.append(n_str.split("__", 1)[-1] if "__" in n_str else n_str)
                self.feature_names[code] = cleaned_names
            else:
                self.feature_names[code] = ALL_FEATURES

            if include_shap:
                return self._init_shap(code)
            
            return True
        except Exception as e:
            print(f"Error loading model {code}: {e}")
            return False

    def _init_shap(self, code):
        """Initializes the SHAP explainer for an already loaded model."""
        try:
            pipeline = self.pipelines.get(code)
            if not pipeline:
                return False
            
            model = pipeline
            if hasattr(pipeline, "named_steps"):
                model = pipeline.named_steps.get('model', pipeline)
                if model == pipeline:
                    for step in reversed(list(pipeline.named_steps.values())):
                        if hasattr(step, "get_feature_importance") or "catboost" in str(type(step)).lower():
                            model = step
                            break
            
            print(f"Initializing SHAP explainer for {code}...")
            self.explainers[code] = shap.TreeExplainer(model)
            print(f"SHAP explainer for {code} ready.")
            return True
        except Exception as e:
            print(f"SHAP initialization failed for {code}: {e}")
            return False

    def get_available_codes(self):
        """Returns a list of available goods codes."""
        return self.available_codes

    def predict(self, input_data: dict, goods_code: str, tolerance: float = 0.15, include_shap: bool = True) -> dict:
        """
        Predicts unit price based on input data and calculates feature contributions (SHAP).
        Args:
            input_data (dict): Dictionary containing input features.
            goods_code (str): The HS code of the goods.   
            include_shap (bool): Whether to calculate feature contributions.
        Returns:
            dict: Contains 'predicted_price', 'lower_bound', 'upper_bound', and optionally 'feature_importance'.
        """
        # The input array/df order must match ALL_FEATURES from constants
        try:
            row = [input_data.get(feature) for feature in ALL_FEATURES]
        except KeyError as e:
             return {"error": f"Missing feature: {e}"}

        input_array = np.array([row], dtype=object)
        input_df = pd.DataFrame(data=input_array, columns=ALL_FEATURES)

        if not self._load_single_model(goods_code, include_shap=include_shap):
            return {"error": f"No model found for code {goods_code}"}

        pipeline = self.pipelines.get(goods_code)
        explainer = self.explainers.get(goods_code)
        preprocessor = self.preprocessors.get(goods_code)
        feature_names = self.feature_names.get(goods_code, ALL_FEATURES)

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

                # Map values to the appropriate feature names (ensure string keys)
                full_importance = {
                    str(feature): float(val) 
                    for feature, val in zip(final_names, values)
                }
                
                # Sort by absolute importance (descending) and take top 5
                sorted_items = sorted(full_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                feature_importance = dict(sorted_items[:5])
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
                        
                    full_importance = {f: float(i) for f, i in zip(feature_names, fi)}
                    
                    # Sort by absolute importance (descending) and take top 10
                    sorted_items = sorted(full_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                    feature_importance = dict(sorted_items[:5])
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
            code_str = str(code)
            if not self._load_single_model(code_str, include_shap=False):
                continue

            pipeline = self.pipelines.get(code_str)
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
