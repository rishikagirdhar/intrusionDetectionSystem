import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

KDD_PATH = Path("archive (1)/kddcup.data_10_percent_corrected")

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

def load_components():
    """Load all model components silently"""
    try:
        model1 = joblib.load('models/model1_binary.pkl')
        model2 = joblib.load('models/model2_attack_type.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        selector = joblib.load('models/feature_selector.pkl')
        feature_info = joblib.load('models/feature_info.pkl')
        return model1, model2, scaler, encoders, selector, feature_info
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        exit()

MODEL1, MODEL2, SCALER, ENCODERS, SELECTOR, FEATURE_INFO = load_components()
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
        if feature in ENCODERS:
            encoder = ENCODERS[feature]
            encoded_values = []
            for x in df_processed[feature]:
                if x in encoder.classes_:
                    encoded_values.append(encoder.transform([x])[0])
                else:
                    encoded_values.append(encoder.transform([encoder.classes_[0]])[0])
            df_processed[feature] = encoded_values
    
    return SCALER.transform(df_processed.values)

def analyze_connection(connection_data):
    """Main detection function - returns action and classification"""
    ip = connection_data.pop('src_ip', 'unknown')
    
    try:
        features_df = prepare_features(connection_data)
        processed_features = preprocess_features(features_df)
        
        binary_proba = MODEL1.predict_proba(SELECTOR.transform(processed_features))[0]
        attack_confidence = float(binary_proba[1])
        normal_confidence = float(binary_proba[0])
        
        attack_proba = MODEL2.predict_proba(processed_features)[0]
        attack_type = MODEL2.classes_[np.argmax(attack_proba)]
        attack_type_confidence = float(np.max(attack_proba))
        
        threshold = ATTACK_THRESHOLDS.get(attack_type, ATTACK_THRESHOLDS['default'])
        risk_level = RISK_LEVELS.get(attack_type, 'MEDIUM')
        
        if attack_type == 'normal':
            if attack_type_confidence > threshold:
                action = 'ALLOW' if normal_confidence > 0.5 else 'MONITOR'
                classification = 'normal'
            else:
                action = 'MONITOR'
                classification = 'normal' if normal_confidence > 0.6 else 'suspicious'
        else:
            if attack_type_confidence > threshold:
                if risk_level == 'CRITICAL':
                    action = 'BLOCK'
                elif risk_level == 'HIGH' and attack_type_confidence > 0.8:
                    action = 'BLOCK'
                elif risk_level == 'HIGH':
                    action = 'ALERT_ADMIN'
                else:
                    action = 'ALERT_ADMIN'
                classification = attack_type
            else:
                action = 'MONITOR'
                classification = 'suspicious'
        
        return {
            'action': action,
            'classification': classification,
            'confidence': attack_type_confidence,
            'risk_level': risk_level
        }
        
    except Exception as e:
        return {'action': 'ERROR', 'classification': 'error', 'error': str(e)}

def evaluate_model(X_test, y_test, num_samples=100):  # Changed default to 100
    """Test the model on sample data"""
    test_indices = []
    for label in y_test.unique():
        label_indices = y_test[y_test == label].index
        sample_size = min(10, len(label_indices))  # Increased from 5 to 10 per class
        test_indices.extend(label_indices[:sample_size])
    
    test_indices = test_indices[:num_samples]
    
    results = []
    for idx in test_indices:
        sample = X_test.loc[idx].to_dict()
        result = analyze_connection(sample)
        results.append({
            'true': y_test.loc[idx],
            'predicted': result['classification'],
            'action': result['action'],
            'confidence': result['confidence'],
            'risk_level': result['risk_level']
        })
    
    return results

def calculate_metrics(results):
    """Calculate performance metrics"""
    exact_acc = sum(1 for r in results if r['true'] == r['predicted']) / len(results)
    binary_acc = sum(1 for r in results if (r['true'] == 'normal') == (r['predicted'] == 'normal')) / len(results)
    
    true_attacks = sum(1 for r in results if r['true'] != 'normal')
    detected_attacks = sum(1 for r in results if r['true'] != 'normal' and r['predicted'] != 'normal')
    
    true_normals = sum(1 for r in results if r['true'] == 'normal')
    false_positives = sum(1 for r in results if r['true'] == 'normal' and r['predicted'] != 'normal')
    
    detection_rate = detected_attacks / true_attacks if true_attacks > 0 else 0
    false_positive_rate = false_positives / true_normals if true_normals > 0 else 0
    
    return {
        'exact_accuracy': exact_acc,
        'binary_accuracy': binary_acc,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'total_samples': len(results)
    }

if __name__ == "__main__":
    df = pd.read_csv(KDD_PATH, names=ALL_FEATURES+['label'], header=None)
    df['label'] = df['label'].str.strip('.')
    X, y = df.drop(columns=['label']), df['label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Changed to evaluate 100 samples
    results = evaluate_model(X_test, y_test, num_samples=100)
    metrics = calculate_metrics(results)
    
    print(" IDS Evaluation Results (100 samples):")
    print(f"   Accuracy: {metrics['exact_accuracy']:.1%}")
    print(f"   Detection Rate: {metrics['detection_rate']:.1%}")
    print(f"   False Positive Rate: {metrics['false_positive_rate']:.1%}")
    print(f"   Total Samples Evaluated: {metrics['total_samples']}")
    
    action_counts = {}
    for r in results:
        action = r['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("\n  Action Summary:")
    for action, count in action_counts.items():
        print(f"   {action}: {count} samples")
    
    print("\n System ready for deployment" if metrics['exact_accuracy'] > 0.9 else "\n  Consider threshold adjustment")