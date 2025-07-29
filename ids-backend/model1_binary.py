import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
import os
from pathlib import Path

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

print("🚀 Starting Model 1 Training (Binary Classification: Normal vs Attack)")
print("=" * 70)

# Full list: 41 feature names + 1 label
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

file_path = Path("archive (1)/kddcup.data_10_percent_corrected")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {os.path.abspath(file_path)}")

print("📂 Loading dataset...")
df = pd.read_csv(file_path, names=column_names, header=None)
df['label'] = df['label'].str.strip('.')

print(f"✅ Dataset shape: {df.shape}")  
print(f"✅ Features: {len(column_names)-1}")

# Create binary labels
df['binary_label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Check class distribution
print(f"\n📊 Binary label distribution:")
binary_dist = df['binary_label'].value_counts()
print(binary_dist)

print(f"\n📊 Original label distribution (top 10):")
print(df['label'].value_counts().head(10))

# Encode categorical features BEFORE creating final dataset
categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}

print(f"\n🔤 Encoding categorical features: {categorical_cols}")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"  ✅ {col}: {len(le.classes_)} unique values")

# Prepare features and target
X = df.drop(columns=['label', 'binary_label'])
y_binary = df['binary_label']

print(f"\n🔧 Feature matrix shape: {X.shape}")
print(f"🎯 Target shape: {y_binary.shape}")

# Scale features using StandardScaler
print(f"\n📏 Applying StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("✅ Feature scaling completed")

# Train-test split for binary classification
print(f"\n🔄 Creating train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"✅ Training set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")
print(f"✅ Training labels: {y_train.value_counts().to_dict()}")

# Feature selection for binary classification
print(f"\n🎯 Performing feature selection...")
selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"✅ Selected {len(selected_features)} features: {list(selected_features)}")

# Convert to DataFrames
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

# Balance the dataset for better binary classification
print(f"\n⚖️ Balancing dataset...")
train_df = X_train_selected_df.copy()
train_df['label'] = y_train.reset_index(drop=True)

normal_df = train_df[train_df['label'] == 'normal']
attack_df = train_df[train_df['label'] == 'attack']

print(f"📊 Before balancing - Normal: {len(normal_df)}, Attack: {len(attack_df)}")

# Smart balancing: don't undersample too much
max_samples_per_class = min(50000, max(len(normal_df), len(attack_df)))
target_samples = min(max_samples_per_class, min(len(normal_df), len(attack_df)))

normal_sampled = normal_df.sample(n=target_samples, random_state=42)
attack_sampled = attack_df.sample(n=target_samples, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([normal_sampled, attack_sampled])
balanced_df = shuffle(balanced_df, random_state=42)

# Final balanced data
X_train_balanced = balanced_df.drop(columns=['label'])
y_train_balanced = balanced_df['label']

print(f"📊 After balancing - {y_train_balanced.value_counts().to_dict()}")
print(f"✅ Balanced dataset shape: {X_train_balanced.shape}")

# Train Random Forest model for binary classification
print(f"\n🤖 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced',
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)

rf_model.fit(X_train_balanced, y_train_balanced)
print("✅ Binary classification model trained successfully!")

# Evaluate model
print(f"\n📊 Evaluating model...")
y_pred = rf_model.predict(X_test_selected_df)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"📊 Binary Classification Accuracy: {acc:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\n🧮 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
print(f"\n🔍 Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Test model predictions
print(f"\n🧪 Testing model predictions...")
test_proba = rf_model.predict_proba(X_test_selected_df[:5])
test_pred = rf_model.predict(X_test_selected_df[:5])

print(f"Model classes: {rf_model.classes_}")
for i in range(5):
    true_label = y_test.iloc[i]
    pred_label = test_pred[i]
    proba = test_proba[i]
    print(f"  Sample {i+1}: True={true_label}, Pred={pred_label}, Proba={proba}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save Model 1 (Binary Classification) components
print(f"\n💾 Saving Model 1 components...")
save_model_safely(rf_model, 'models/model1_binary.pkl')
save_model_safely(scaler, 'models/scaler.pkl')
save_model_safely(encoders, 'models/encoders.pkl')
save_model_safely(selector, 'models/feature_selector.pkl')

# Save training data for Model 2 (multiclass)
print(f"\n💾 Preparing data for Model 2...")
final_data = df.copy()
final_data['attack_type'] = df['label']  # Keep original attack types
final_data.to_csv('final_model_data.csv', index=False)

# Save feature information for consistency
feature_info = {
    'all_features': list(X.columns),
    'selected_features': list(selected_features),
    'categorical_features': categorical_cols,
    'numerical_features': [col for col in X.columns if col not in categorical_cols],
    'model1_classes': list(rf_model.classes_),
    'scaler_type': 'StandardScaler'
}
save_model_safely(feature_info, 'models/feature_info.pkl')

print(f"\n✅ Saved all Model 1 files successfully:")
print("   - models/model1_binary.pkl")
print("   - models/scaler.pkl") 
print("   - models/encoders.pkl")
print("   - models/feature_selector.pkl")
print("   - models/feature_info.pkl")
print("   - final_model_data.csv")

# Save model metadata
model1_info = {
    'model_type': 'RandomForestClassifier',
    'accuracy': acc,
    'classes': list(rf_model.classes_),
    'num_features': len(selected_features),
    'selected_features': list(selected_features),
    'training_samples': len(X_train_balanced),
    'test_samples': len(X_test_selected_df),
    'class_distribution': dict(y_train_balanced.value_counts()),
    'feature_selection_k': 15,
    'balancing_strategy': 'equal_sampling'
}
save_model_safely(model1_info, 'models/model1_info.pkl')

print(f"\n🎉 Model 1 training completed successfully!")
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Classes: {list(rf_model.classes_)}")
print(f"✅ Ready for Model 2 training!")
print("✅ Run model2_multiclass.py next")