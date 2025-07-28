import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def preprocess_input(raw_data, scaler, encoders):
    """
    Preprocess raw network connection data for model prediction.
    Handles missing features, maintains feature order, and ensures proper data types.
    
    Args:
        raw_data (dict): Input connection features
        scaler: Fitted scaler object
        encoders: Dictionary of fitted label encoders
        
    Returns:
        pd.DataFrame: Processed features with preserved column names
    
    Raises:
        ValueError: If critical features are missing or data cannot be processed
    """
    # 1. Validate input
    if not isinstance(raw_data, dict):
        raise ValueError("Input must be a dictionary of features")
    
    # 2. Handle missing features
    expected_features = scaler.feature_names_in_
    df = pd.DataFrame(columns=expected_features)
    
    # 3. Fill data with proper type handling
    for feat in expected_features:
        try:
            # Handle present features
            if feat in raw_data:
                df[feat] = [raw_data[feat]]
            # Handle missing features
            else:
                fill_value = 0 if feat in NUMERICAL_FEATURES else -1
                df[feat] = [fill_value]
                print(f"⚠️ Warning: Filled missing feature '{feat}' with {fill_value}")
        except Exception as e:
            raise ValueError(f"Error processing feature '{feat}': {str(e)}")
    
    # 4. Encode categorical features with safety checks
    for col in CATEGORICAL_FEATURES:
        if col not in encoders:
            raise ValueError(f"Missing encoder for categorical feature: {col}")
        
        # Convert to string and handle unseen categories
        df[col] = df[col].astype(str).apply(
            lambda x: x if x in encoders[col].classes_ else "UNKNOWN"
        )
        df[col] = encoders[col].transform(df[col])
    
    # 5. Final validation and scaling
    try:
        # Ensure numerical stability
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        scaled_data = scaler.transform(df)
        return pd.DataFrame(scaled_data, columns=expected_features)
    except Exception as e:
        raise ValueError(f"Scaling failed: {str(e)}")
def load_preprocessors():
    """Load saved preprocessing objects with validation"""
    import joblib
    try:
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        
        # Verify we have all required encoders
        for col in CATEGORICAL_FEATURES:
            if col not in encoders:
                raise ValueError(f"Missing encoder for categorical feature: {col}")
                
        return scaler, encoders
    except FileNotFoundError as e:
        raise Exception(f"Preprocessor files not found: {str(e)}. Train models first.")