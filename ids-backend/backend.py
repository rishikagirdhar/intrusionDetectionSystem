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

MODEL1 = joblib.load('models/model1_binary.pkl')
MODEL2 = joblib.load('models/model2_attack_type.pkl')
SCALER = joblib.load('models/scaler.pkl')
ENCODERS = joblib.load('models/encoders.pkl')
SELECTOR = joblib.load('models/feature_selector.pkl')
FEATURE_INFO = joblib.load('models/feature_info.pkl')

ATTACK_THRESHOLDS = {
    'normal': 0.3, 'neptune': 0.45, 'smurf': 0.45, 'back': 0.55, 'teardrop': 0.55,
    'nmap': 0.6, 'satan': 0.6, 'ipsweep': 0.6, 'portsweep': 0.6, 'warezclient': 0.65,
    'warezmaster': 0.65, 'ftp_write': 0.7, 'guess_passwd': 0.7, 'imap': 0.7,
    'multihop': 0.7, 'phf': 0.7, 'spy': 0.7, 'buffer_overflow': 0.8,
    'loadmodule': 0.8, 'perl': 0.8, 'rootkit': 0.8, 'default': 0.6
}

RISK_LEVELS = {
    'normal': 'LOW', 'neptune': 'HIGH', 'smurf': 'HIGH', 'back': 'HIGH',
    'nmap': 'MEDIUM', 'satan': 'MEDIUM', 'ipsweep': 'MEDIUM', 'portsweep': 'MEDIUM',
    'warezclient': 'HIGH', 'warezmaster': 'HIGH', 'ftp_write': 'CRITICAL',
    'guess_passwd': 'CRITICAL', 'imap': 'CRITICAL', 'multihop': 'CRITICAL',
    'phf': 'CRITICAL', 'spy': 'CRITICAL', 'buffer_overflow': 'CRITICAL',
    'loadmodule': 'CRITICAL', 'perl': 'CRITICAL', 'rootkit': 'CRITICAL'
}


ALL_FEATURES = FEATURE_INFO['all_features']
CATEGORICAL_FEATURES = FEATURE_INFO['categorical_features']
NUMERICAL_FEATURES = FEATURE_INFO['numerical_features']

def prepare_features(features_dict):
    """Convert input dictionary to properly formatted DataFrame"""
    df = pd.DataFrame([features_dict])
    
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            if feature in NUMERICAL_FEATURES:
                df[feature] = 0.0
            else:
                df[feature] = 'tcp' if feature == 'protocol_type' else \
                              'http' if feature == 'service' else 'SF'
    
    for feature in NUMERICAL_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)
    
    return df[ALL_FEATURES]

def preprocess_features(df):
    """Apply encoding and scaling to features"""
    df_processed = df.copy()
    
    for feature in CATEGORICAL_FEATURES:
        encoder = ENCODERS[feature]
        encoded_values = []
        for x in df_processed[feature]:
            if x in encoder.classes_:
                encoded_values.append(encoder.transform([x])[0])
            else:
                encoded_values.append(encoder.transform([encoder.classes_[0]])[0])
        df_processed[feature] = encoded_values
    
    return SCALER.transform(df_processed.values)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for intrusion detection predictions"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        
        
        required_fields = ['protocol_type', 'service', 'src_bytes', 'dst_bytes', 'flag']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}'
            }), 400
        
        
        features_df = prepare_features(data)
        processed_features = preprocess_features(features_df)
        selected_features = SELECTOR.transform(processed_features)
        
       
        binary_pred = MODEL1.predict(selected_features)[0]
        attack_proba = MODEL2.predict_proba(processed_features)[0]
        attack_type = MODEL2.classes_[np.argmax(attack_proba)]
        confidence = np.max(attack_proba)
        
        
        threshold = ATTACK_THRESHOLDS.get(attack_type, ATTACK_THRESHOLDS['default'])
        if confidence < threshold:
            attack_type = f"suspicious ({attack_type})"
        
        return jsonify({
            'status': 'success',
            'binary_prediction': 'attack' if binary_pred == 1 else 'normal',
            'attack_type': attack_type,
            'confidence': float(confidence),
            'risk_level': RISK_LEVELS.get(attack_type.split('(')[0], 'MEDIUM'),
            'threshold': float(threshold)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def home():
    return "IDS Model API - Send POST requests to /predict"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({'error': 'No messages provided'}), 400

        # Initialize Groq (API key from .env)
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")  # Key is secure!
        )
        
        # Get response from Groq
        response = llm.invoke(data['messages'])
        return jsonify({'response': response.content})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)