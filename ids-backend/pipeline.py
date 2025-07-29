import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

KDD_PATH = Path("archive (1)/kddcup.data_10_percent_corrected")

# Updated thresholds for binary classification (Model 1)
BINARY_THRESHOLD = 0.5  # Threshold for normal vs attack classification

# Attack-specific thresholds for Model 2 (when attack is detected)
ATTACK_THRESHOLDS = {
    'neptune': 0.45, 'smurf': 0.45, 'back': 0.55, 'teardrop': 0.55,
    'nmap': 0.6, 'satan': 0.6, 'ipsweep': 0.6, 'portsweep': 0.6, 'warezclient': 0.65,
    'warezmaster': 0.65, 'ftp_write': 0.7, 'guess_passwd': 0.7, 'imap': 0.7,
    'multihop': 0.7, 'phf': 0.7, 'spy': 0.7, 'buffer_overflow': 0.8,
    'loadmodule': 0.8, 'perl': 0.8, 'rootkit': 0.8, 'land': 0.8, 'pod': 0.6,
    'default': 0.6
}

RISK_LEVELS = {
    'neptune': 'HIGH', 'smurf': 'HIGH', 'back': 'HIGH',
    'nmap': 'MEDIUM', 'satan': 'MEDIUM', 'ipsweep': 'MEDIUM', 'portsweep': 'MEDIUM',
    'warezclient': 'HIGH', 'warezmaster': 'HIGH', 'ftp_write': 'CRITICAL',
    'guess_passwd': 'CRITICAL', 'imap': 'CRITICAL', 'multihop': 'CRITICAL',
    'phf': 'CRITICAL', 'spy': 'CRITICAL', 'buffer_overflow': 'CRITICAL',
    'loadmodule': 'CRITICAL', 'perl': 'CRITICAL', 'rootkit': 'CRITICAL',
    'land': 'CRITICAL', 'pod': 'MEDIUM', 'teardrop': 'HIGH'
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
        
        print(f"✅ Loaded Model 1 (Binary): {len(model1.classes_)} classes")
        print(f"✅ Model 1 classes: {list(model1.classes_)}")  # DEBUG: Show actual classes
        print(f"✅ Loaded Model 2 (Attack): {len(model2.classes_)} classes")
        print(f"✅ Model 2 classes: {list(model2.classes_)}")
        
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

def analyze_connection(connection_data, debug=False):
    """Main detection function using two-model architecture - FIXED VERSION"""
    ip = connection_data.pop('src_ip', 'unknown')
    
    try:
        # Step 1: Prepare and preprocess features
        features_df = prepare_features(connection_data)
        processed_features = preprocess_features(features_df)
        
        # Step 2: Binary classification (Model 1) - Normal vs Attack
        selected_features = SELECTOR.transform(processed_features)
        binary_proba = MODEL1.predict_proba(selected_features)[0]
        binary_pred = MODEL1.predict(selected_features)[0]
        
        # FIXED: Get correct class indices
        model1_classes = list(MODEL1.classes_)
        
        # Find the correct indices for normal and attack
        if 'normal' in model1_classes:
            normal_idx = model1_classes.index('normal')
            attack_idx = model1_classes.index('attack') if 'attack' in model1_classes else (1 - normal_idx)
        elif 'attack' in model1_classes:
            attack_idx = model1_classes.index('attack')
            normal_idx = 1 - attack_idx
        else:
            # Fallback - assume first class is normal, second is attack
            normal_idx = 0
            attack_idx = 1
        
        normal_confidence = float(binary_proba[normal_idx])
        attack_confidence = float(binary_proba[attack_idx])
        
        if debug:
            print(f"  DEBUG: Model1 classes: {model1_classes}")
            print(f"  DEBUG: Normal idx: {normal_idx}, Attack idx: {attack_idx}")
            print(f"  DEBUG: Binary prediction: {binary_pred}")
            print(f"  DEBUG: Normal conf: {normal_confidence:.3f}, Attack conf: {attack_confidence:.3f}")
        
        # Step 3: Decision based on binary classification
        # Use the actual binary prediction from the model, not just confidence
        if binary_pred == 'normal' or (binary_pred == model1_classes[normal_idx]):
            # Classified as NORMAL by Model 1
            return {
                'action': 'ALLOW',
                'classification': 'normal',
                'confidence': normal_confidence,
                'risk_level': 'LOW',
                'model_used': 'Model1_Normal',
                'debug_info': {
                    'binary_pred': binary_pred,
                    'normal_conf': normal_confidence,
                    'attack_conf': attack_confidence
                } if debug else None
            }
        
        # Step 4: If attack detected, use Model 2 for attack type classification
        attack_proba = MODEL2.predict_proba(processed_features)[0]
        attack_type = MODEL2.classes_[np.argmax(attack_proba)]
        attack_type_confidence = float(np.max(attack_proba))
        
        # Step 5: Apply attack-specific threshold
        threshold = ATTACK_THRESHOLDS.get(attack_type, ATTACK_THRESHOLDS['default'])
        risk_level = RISK_LEVELS.get(attack_type, 'MEDIUM')
        
        if debug:
            print(f"  DEBUG: Attack type: {attack_type}")
            print(f"  DEBUG: Attack type confidence: {attack_type_confidence:.3f}")
            print(f"  DEBUG: Threshold: {threshold}")
        
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
            
            classification = attack_type
        else:
            # Low confidence - treat as suspicious
            action = 'MONITOR'
            classification = 'suspicious'
        
        return {
            'action': action,
            'classification': classification,
            'confidence': attack_type_confidence,
            'risk_level': risk_level,
            'model_used': f'Model2_{attack_type}',
            'binary_confidence': attack_confidence,
            'debug_info': {
                'binary_pred': binary_pred,
                'normal_conf': normal_confidence,
                'attack_conf': attack_confidence,
                'attack_type': attack_type,
                'attack_threshold': threshold
            } if debug else None
        }
        
    except Exception as e:
        return {
            'action': 'ERROR', 
            'classification': 'error', 
            'confidence': 0.0,
            'risk_level': 'UNKNOWN',
            'error': str(e)
        }

def evaluate_model(X_test, y_test, num_samples=100, debug_samples=5):
    """Test the model on sample data with debugging"""
    # Get balanced sample across all classes
    test_indices = []
    unique_labels = y_test.unique()
    samples_per_class = max(1, num_samples // len(unique_labels))
    
    for label in unique_labels:
        label_indices = y_test[y_test == label].index.tolist()
        sample_size = min(samples_per_class, len(label_indices))
        test_indices.extend(label_indices[:sample_size])
    
    # If we need more samples, add randomly
    if len(test_indices) < num_samples:
        remaining_indices = [idx for idx in y_test.index if idx not in test_indices]
        additional_needed = num_samples - len(test_indices)
        test_indices.extend(remaining_indices[:additional_needed])
    
    test_indices = test_indices[:num_samples]
    
    results = []
    debug_count = 0
    
    for idx in test_indices:
        sample = X_test.loc[idx].to_dict()
        
        # Debug first few samples
        should_debug = debug_count < debug_samples
        if should_debug:
            print(f"\n🔍 DEBUG Sample {debug_count + 1} (True: {y_test.loc[idx]}):")
            debug_count += 1
        
        result = analyze_connection(sample, debug=should_debug)
        
        true_label = y_test.loc[idx]
        
        results.append({
            'true': true_label,
            'predicted': result['classification'],
            'action': result['action'],
            'confidence': result['confidence'],
            'risk_level': result['risk_level'],
            'model_used': result.get('model_used', 'unknown')
        })
        
        if should_debug:
            print(f"  Result: {result['classification']} (Action: {result['action']})")
    
    return results

def calculate_metrics(results):
    """Calculate performance metrics"""
    total = len(results)
    
    # Exact accuracy
    exact_correct = sum(1 for r in results if r['true'] == r['predicted'])
    exact_acc = exact_correct / total
    
    # Binary accuracy (normal vs attack)
    binary_correct = sum(1 for r in results if (r['true'] == 'normal') == (r['predicted'] == 'normal'))
    binary_acc = binary_correct / total
    
    # Attack detection metrics
    true_attacks = sum(1 for r in results if r['true'] != 'normal')
    detected_attacks = sum(1 for r in results if r['true'] != 'normal' and r['predicted'] != 'normal')
    
    # False positive metrics
    true_normals = sum(1 for r in results if r['true'] == 'normal')
    false_positives = sum(1 for r in results if r['true'] == 'normal' and r['predicted'] != 'normal')
    
    detection_rate = detected_attacks / true_attacks if true_attacks > 0 else 0
    false_positive_rate = false_positives / true_normals if true_normals > 0 else 0
    
    return {
        'exact_accuracy': exact_acc,
        'binary_accuracy': binary_acc,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'total_samples': total,
        'true_attacks': true_attacks,
        'true_normals': true_normals,
        'detected_attacks': detected_attacks,
        'false_positives': false_positives
    }

def print_detailed_results(results, metrics):
    """Print detailed analysis of results"""
    print(f"\n🔍 Detailed Analysis:")
    print(f"   Total Samples: {metrics['total_samples']}")
    print(f"   True Normal: {metrics['true_normals']}")
    print(f"   True Attacks: {metrics['true_attacks']}")
    print(f"   Detected Attacks: {metrics['detected_attacks']}")
    print(f"   False Positives: {metrics['false_positives']}")
    
    print(f"\n📊 Model Usage:")
    model_usage = {}
    for r in results:
        model = r.get('model_used', 'unknown')
        model_usage[model] = model_usage.get(model, 0) + 1
    
    for model, count in sorted(model_usage.items()):
        print(f"   {model}: {count} samples")
    
    # Show some example classifications
    print(f"\n🎯 Example Classifications:")
    normal_examples = [r for r in results if r['true'] == 'normal'][:3]
    attack_examples = [r for r in results if r['true'] != 'normal'][:3]
    
    for r in normal_examples:
        status = "✅" if r['predicted'] == 'normal' else "❌"
        print(f"   {status} Normal → {r['predicted']} (Action: {r['action']}, Conf: {r['confidence']:.3f})")
    
    for r in attack_examples:
        status = "✅" if r['predicted'] == r['true'] else "❌"
        print(f"   {status} {r['true']} → {r['predicted']} (Action: {r['action']}, Conf: {r['confidence']:.3f})")

if __name__ == "__main__":
    print("🚀 Starting IDS Pipeline Evaluation (FIXED VERSION)...")
    
    # Load test data
    df = pd.read_csv(KDD_PATH, names=ALL_FEATURES+['label'], header=None)
    df['label'] = df['label'].str.strip('.')
    X, y = df.drop(columns=['label']), df['label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📊 Test data distribution:")
    print(y_test.value_counts().head(10))
    
    # Evaluate model with debugging
    print(f"\n🔧 Running evaluation with debugging...")
    results = evaluate_model(X_test, y_test, num_samples=100, debug_samples=3)
    metrics = calculate_metrics(results)
    
    print(f"\n🎯 IDS Evaluation Results ({metrics['total_samples']} samples):")
    print(f"   Exact Accuracy: {metrics['exact_accuracy']:.1%}")
    print(f"   Binary Accuracy: {metrics['binary_accuracy']:.1%}")
    print(f"   Attack Detection Rate: {metrics['detection_rate']:.1%}")
    print(f"   False Positive Rate: {metrics['false_positive_rate']:.1%}")
    
    # Action summary
    action_counts = {}
    for r in results:
        action = r['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"\n⚡ Action Summary:")
    for action, count in sorted(action_counts.items()):
        print(f"   {action}: {count} samples ({count/len(results):.1%})")
    
    # Print detailed results
    print_detailed_results(results, metrics)
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if metrics['false_positive_rate'] > 0.5:
        print(f"   - High false positive rate detected")
        print(f"   - Check Model 1 class mapping")
        print(f"   - Consider adjusting BINARY_THRESHOLD (current: {BINARY_THRESHOLD})")
    
    if metrics['detection_rate'] < 0.5:
        print(f"   - Low attack detection rate")
        print(f"   - Model 1 may be too conservative")
        print(f"   - Consider retraining with different parameters")
    
    # System status
    if metrics['binary_accuracy'] > 0.8 and metrics['false_positive_rate'] < 0.2:
        print(f"\n✅ System ready for deployment!")
    else:
        print(f"\n🔧 System needs tuning before deployment")