import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def save_model_safely(obj, path):
    """Atomic save to prevent corruption"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.temp"
    try:
        with open(temp_path, 'wb') as f:
            joblib.dump(obj, f)
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to save {path}: {str(e)}")

# 1. Load the cleaned dataset
print("Loading dataset...")
df = pd.read_csv('final_model_data.csv')

# 2. Separate features and target (focus on attack types)
X = df.drop(columns=['attack_type', 'label', 'binary_label'])
y = df['attack_type']  # Multi-class target

print(f"Dataset shape: {X.shape}")
print(f"Attack type distribution:\n{y.value_counts()}")

# 3. Load existing encoders to ensure consistency
print("Loading existing encoders...")
try:
    existing_encoders = joblib.load('models/encoders.pkl')
    print("✅ Loaded existing encoders")
except:
    print("⚠️ No existing encoders found, creating new ones")
    existing_encoders = {}

# 4. Encode categorical features using existing encoders
categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}

for col in categorical_cols:
    if col in existing_encoders:
        # Use existing encoder
        encoders[col] = existing_encoders[col]
        # Transform using existing encoder
        unique_vals = X[col].unique()
        known_vals = existing_encoders[col].classes_
        
        # Handle unknown categories
        X[col] = X[col].apply(lambda x: x if x in known_vals else known_vals[0])
        X[col] = existing_encoders[col].transform(X[col])
    else:
        # Create new encoder
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

print("✅ Categorical features encoded")

# 5. Feature scaling using same scaler as Model 1
print("Loading existing scaler...")
try:
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Loaded existing scaler")
except:
    print("⚠️ No existing scaler found, creating new one")
    scaler = StandardScaler()
    scaler.fit(X)

# Apply scaling
X_scaled = scaler.transform(X)

# 6. Train-test split (80-20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 7. Define models to evaluate
models = {
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=20),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

# 8. Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"{name} Test Accuracy: {acc:.4f}")
    print(f"{name} Weighted F1-score: {report['weighted avg']['f1-score']:.4f}")
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'report': report
    }

# 9. Select best model (prioritizing F1-score for minority classes)
best_name = max(results, key=lambda k: results[k]['report']['weighted avg']['f1-score'])
best_model = results[best_name]['model']
best_accuracy = results[best_name]['accuracy']
best_f1 = results[best_name]['report']['weighted avg']['f1-score']

print(f"\n🏆 Best model: {best_name}")
print(f"📊 Accuracy: {best_accuracy:.4f}")
print(f"📊 Weighted F1-score: {best_f1:.4f}")

# Show detailed classification report for best model
print(f"\n📋 Detailed Classification Report for {best_name}:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# 10. Save Model 2 components
print("\nSaving Model 2 components...")
save_model_safely(best_model, 'models/model2_attack_type.pkl')

# Save model metadata
model_info = {
    'model_name': best_name,
    'accuracy': best_accuracy,
    'f1_score': best_f1,
    'classes': list(best_model.classes_),
    'feature_names': list(df.drop(columns=['attack_type', 'label', 'binary_label']).columns)
}
save_model_safely(model_info, 'models/model2_info.pkl')

print("\n✅ Saved Model 2 files successfully:")
print("- models/model2_attack_type.pkl")
print("- models/model2_info.pkl")

# 11. Test model loading
print("\nTesting model loading...")
try:
    loaded_model = joblib.load('models/model2_attack_type.pkl')
    print("✅ Model 2 loaded successfully")
    print(f"Model classes: {loaded_model.classes_}")
except Exception as e:
    print(f"❌ Error loading Model 2: {e}")

print("\n✅ Model 2 training completed successfully!")