from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq  
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app) 
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY is not set. Please add it to your .env file.")

print("Loading IDS Models...")

# Load models and components
try:
    MODEL1 = joblib.load('models/model1_binary.pkl')
    MODEL2 = joblib.load('models/model2_attack_type.pkl')
    SCALER = joblib.load('models/scaler.pkl')
    ENCODERS = joblib.load('models/encoders.pkl')
    SELECTOR = joblib.load('models/feature_selector.pkl')
    FEATURE_INFO = joblib.load('models/feature_info.pkl')
    
    print(f"Model 1 (Binary): {MODEL1.classes_}")
    print(f"Model 2 (Attack): {len(MODEL2.classes_)} attack types")
    print(f"Feature info loaded: {len(FEATURE_INFO['all_features'])} features")
    
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Configuration for decision making
BINARY_THRESHOLD = 0.5  # Threshold for normal vs attack (Model 1)

# Attack-specific thresholds for Model 2
ATTACK_THRESHOLDS = {
    'neptune': 0.45, 'smurf': 0.45, 'back': 0.55, 'teardrop': 0.55,
    'nmap': 0.6, 'satan': 0.6, 'ipsweep': 0.6, 'portsweep': 0.6, 
    'warezclient': 0.65, 'warezmaster': 0.65, 'ftp_write': 0.7, 
    'guess_passwd': 0.7, 'imap': 0.7, 'multihop': 0.7, 'phf': 0.7, 
    'spy': 0.7, 'buffer_overflow': 0.8, 'loadmodule': 0.8, 'perl': 0.8, 
    'rootkit': 0.8, 'land': 0.8, 'pod': 0.6, 'default': 0.6
}

RISK_LEVELS = {
    'neptune': 'HIGH', 'smurf': 'HIGH', 'back': 'HIGH', 'teardrop': 'HIGH',
    'nmap': 'MEDIUM', 'satan': 'MEDIUM', 'ipsweep': 'MEDIUM', 'portsweep': 'MEDIUM',
    'warezclient': 'HIGH', 'warezmaster': 'HIGH', 'ftp_write': 'CRITICAL',
    'guess_passwd': 'CRITICAL', 'imap': 'CRITICAL', 'multihop': 'CRITICAL',
    'phf': 'CRITICAL', 'spy': 'CRITICAL', 'buffer_overflow': 'CRITICAL',
    'loadmodule': 'CRITICAL', 'perl': 'CRITICAL', 'rootkit': 'CRITICAL',
    'land': 'CRITICAL', 'pod': 'MEDIUM'
}

# Extract feature information
ALL_FEATURES = FEATURE_INFO['all_features']
CATEGORICAL_FEATURES = FEATURE_INFO['categorical_features']
NUMERICAL_FEATURES = FEATURE_INFO['numerical_features']

def prepare_features(features_dict):
    """Convert input dictionary to properly formatted DataFrame"""
    df = pd.DataFrame([features_dict])
    
    # Add missing features with appropriate defaults
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            if feature in NUMERICAL_FEATURES:
                df[feature] = 0.0
            else:
                # Categorical feature defaults
                if feature == 'protocol_type':
                    df[feature] = 'tcp'
                elif feature == 'service':
                    df[feature] = 'http'
                elif feature == 'flag':
                    df[feature] = 'SF'
                else:
                    df[feature] = 'tcp'  # fallback
    
    # Convert numerical features
    for feature in NUMERICAL_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)
    
    return df[ALL_FEATURES]

def preprocess_features(df):
    """Apply encoding and scaling to features"""
    df_processed = df.copy()
    
    # Encode categorical features
    for feature in CATEGORICAL_FEATURES:
        if feature in ENCODERS:
            encoder = ENCODERS[feature]
            encoded_values = []
            for x in df_processed[feature]:
                if x in encoder.classes_:
                    encoded_values.append(encoder.transform([x])[0])
                else:
                    # Use the first class as default for unknown values
                    encoded_values.append(encoder.transform([encoder.classes_[0]])[0])
            df_processed[feature] = encoded_values
    
    # Apply scaling
    return SCALER.transform(df_processed.values)
def analyze_connection(connection_data):
    """Two-stage intrusion detection analysis - COMPLETE FIX"""
    try:
        # Step 1: Prepare and preprocess features
        features_df = prepare_features(connection_data)
        processed_features = preprocess_features(features_df)
        
        # Step 2: Binary classification (Model 1) - Normal vs Attack
        selected_features = SELECTOR.transform(processed_features)
        binary_proba = MODEL1.predict_proba(selected_features)[0]
        binary_pred = MODEL1.predict(selected_features)[0]
        
        # CRITICAL FIX: Handle the actual class order ['attack', 'normal']
        model1_classes = list(MODEL1.classes_)
        print(f"DEBUG: Model1 classes: {model1_classes}")  # Temporary debug
        
        # Map based on actual class positions
        if model1_classes == ['attack', 'normal']:
            attack_idx = 0
            normal_idx = 1
        elif model1_classes == ['normal', 'attack']:
            normal_idx = 0
            attack_idx = 1
        else:
            # Dynamic mapping
            normal_idx = model1_classes.index('normal') if 'normal' in model1_classes else 1
            attack_idx = model1_classes.index('attack') if 'attack' in model1_classes else 0
        
        normal_confidence = float(binary_proba[normal_idx])
        attack_confidence = float(binary_proba[attack_idx])
        
        # Debug information
        debug_info = {
            'model1_classes': model1_classes,
            'normal_idx': normal_idx,
            'attack_idx': attack_idx,
            'binary_pred': binary_pred,
            'normal_confidence': normal_confidence,
            'attack_confidence': attack_confidence,
            'binary_proba_raw': [float(x) for x in binary_proba]
        }
        
        # Step 3: Decision logic - Use actual model prediction
        if binary_pred == 'normal':
            # Classified as NORMAL by Model 1
            return {
                'status': 'success',
                'binary_prediction': 'normal',
                'binary_confidence': normal_confidence,
                'final_classification': 'normal',
                'confidence': normal_confidence,
                'risk_level': 'LOW',
                'action': 'ALLOW',
                'threshold_used': 'model_prediction',
                'model_path': 'Model1_Only',
                'details': {
                    'normal_confidence': normal_confidence,
                    'attack_confidence': attack_confidence,
                    'went_to_model2': False
                },
                'debug': debug_info
            }
        
        # Step 4: If attack detected, use Model 2 for attack type classification
        attack_proba = MODEL2.predict_proba(processed_features)[0]
        attack_type = MODEL2.classes_[np.argmax(attack_proba)]
        attack_type_confidence = float(np.max(attack_proba))
        
        # Get top 5 attack predictions
        top_attacks = {}
        top_indices = np.argsort(attack_proba)[-5:][::-1]
        for idx in top_indices:
            attack_name = MODEL2.classes_[idx]
            confidence = float(attack_proba[idx])
            top_attacks[attack_name] = confidence
        
        # Step 5: Apply attack-specific threshold
        threshold = ATTACK_THRESHOLDS.get(attack_type, ATTACK_THRESHOLDS['default'])
        risk_level = RISK_LEVELS.get(attack_type, 'MEDIUM')
        
        # Enhanced debug for Model 2
        debug_info['model2_debug'] = {
            'model2_prediction': attack_type,
            'model2_confidence': attack_type_confidence,
            'threshold_applied': threshold,
            'top_5_attacks': top_attacks,
            'risk_level': risk_level
        }
        
        if attack_type_confidence >= threshold:
            # High confidence attack classification
            if risk_level == 'CRITICAL':
                action = 'BLOCK'
            elif risk_level == 'HIGH' and attack_type_confidence > 0.8:
                action = 'BLOCK'
            elif risk_level == 'HIGH':
                action = 'ALERT_ADMIN'
            else:
                action = 'ALERT_ADMIN'
            
            final_classification = attack_type
            final_confidence = attack_type_confidence
        else:
            # Low confidence - treat as suspicious
            action = 'MONITOR'
            final_classification = f'suspicious ({attack_type})'
            final_confidence = attack_type_confidence
        
        return {
            'status': 'success',
            'binary_prediction': 'attack',
            'binary_confidence': attack_confidence,
            'final_classification': final_classification,
            'confidence': final_confidence,
            'risk_level': risk_level,
            'action': action,
            'threshold_used': threshold,
            'model_path': 'Model1_then_Model2',
            'details': {
                'normal_confidence': normal_confidence,
                'attack_confidence': attack_confidence,
                'went_to_model2': True,
                'attack_type': attack_type,
                'attack_type_confidence': attack_type_confidence,
                'top_5_attacks': top_attacks
            },
            'debug': debug_info
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in analyze_connection: {error_details}")
        
        return {
            'status': 'error',
            'message': str(e),
            'final_classification': 'error',
            'confidence': 0.0,
            'risk_level': 'UNKNOWN',
            'action': 'ERROR',
            'debug': {'error_details': error_details}
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for intrusion detection predictions"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        
        # Check for some basic required fields
        required_fields = ['protocol_type', 'service', 'flag']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Remove IP if present (not used in model)
        connection_data = {k: v for k, v in data.items() if k != 'src_ip'}
        
        # Analyze the connection
        result = analyze_connection(connection_data)
        
        # Add additional metadata
        result['timestamp'] = pd.Timestamp.now().isoformat()
        result['model_info'] = {
            'model1_classes': list(MODEL1.classes_),
            'model2_classes': list(MODEL2.classes_),
            'features_used': len(ALL_FEATURES)
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"ERROR in predict(): {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'final_classification': 'error',
            'confidence': 0.0,
            'risk_level': 'UNKNOWN',
            'action': 'ERROR'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Quick test of models
        test_data = {
            'protocol_type': 'tcp',
            'service': 'http',
            'flag': 'SF',
            'src_bytes': 100,
            'dst_bytes': 200,
            'duration': 1
        }
        
        result = analyze_connection(test_data)
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': True,
            'model1_classes': len(MODEL1.classes_),
            'model2_classes': len(MODEL2.classes_),
            'test_prediction': result['final_classification']
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'IDS Model API',
        'endpoints': {
            '/predict': 'POST - Intrusion detection prediction',
            '/health': 'GET - Health check',
            '/chat': 'POST - Chat with AI assistant'
        },
        'models': {
            'model1': f'Binary classification ({MODEL1.classes_})',
            'model2': f'Attack type classification ({len(MODEL2.classes_)} types)'
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with AI assistant"""
    try:
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({'error': 'No messages provided'}), 400

        # Initialize Groq (API key from .env)
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Get response from Groq
        response = llm.invoke(data['messages'])
        return jsonify({'response': response.content})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting IDS API Server...")
    print(f"Model 1: {MODEL1.__class__.__name__} with {len(MODEL1.classes_)} classes")
    print(f"Model 2: {MODEL2.__class__.__name__} with {len(MODEL2.classes_)} classes")
    print(f"Features: {len(ALL_FEATURES)} total, {len(CATEGORICAL_FEATURES)} categorical")
    print("Server starting on http://0.0.0.0:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)