import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Changed to StandardScaler
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

file_path = Path(r"C:\Users\HP\OneDrive\Desktop\chatbot drdo\archive (1)\kddcup.data_10_percent_corrected")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {os.path.abspath(file_path)}")

df = pd.read_csv(file_path, names=column_names, header=None)
# Clean the label (remove trailing dot)
df['label'] = df['label'].str.strip('.')

print("✅ Shape:", df.shape)  
print(df[['dst_host_srv_rerror_rate', 'label']].head())

# Create binary labels
df['binary_label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Check class distribution
print("Binary label distribution:")
print(df['binary_label'].value_counts())

print("Original label distribution:")
print(df['label'].value_counts())

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("Encoded categorical features:")
print(df[categorical_cols].head())

# Prepare features and target
X = df.drop(columns=['label', 'binary_label'])
y_binary = df['binary_label']
y_multiclass = df['label']

# Scale features using StandardScaler (consistent with Model 2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split for binary classification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Feature selection for binary classification
selector = SelectKBest(score_func=f_classif, k=15)  # Increased from 10
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"Selected features for binary classification: {list(selected_features)}")

# Convert to DataFrames
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

# Balance the dataset
train_df = X_train_selected_df.copy()
train_df['label'] = y_train.reset_index(drop=True)

normal_df = train_df[train_df['label'] == 'normal']
attack_df = train_df[train_df['label'] == 'attack']

# Undersample attack to match normal
min_samples = min(len(normal_df), len(attack_df))
normal_sampled = normal_df.sample(n=min_samples, random_state=42)
attack_sampled = attack_df.sample(n=min_samples, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([normal_sampled, attack_sampled])
balanced_df = shuffle(balanced_df, random_state=42)

# Final balanced data
X_train_balanced = balanced_df.drop(columns=['label'])
y_train_balanced = balanced_df['label']

print("✅ Balanced X shape:", X_train_balanced.shape)
print("✅ Balanced class counts:\n", y_train_balanced.value_counts())

# Train Random Forest model for binary classification
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_balanced, y_train_balanced)

print("✅ Binary classification model trained successfully!")

# Evaluate model
y_pred = rf_model.predict(X_test_selected_df)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"📊 Binary Classification Accuracy: {acc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create models directory
os.makedirs('models', exist_ok=True)

# Save Model 1 (Binary Classification) components
save_model_safely(rf_model, 'models/model1_binary.pkl')
save_model_safely(scaler, 'models/scaler.pkl')
save_model_safely(encoders, 'models/encoders.pkl')
save_model_safely(selector, 'models/feature_selector.pkl')

# Save training data for Model 2 (multiclass)
final_data = df.copy()
final_data['attack_type'] = df['label']
final_data.to_csv('final_model_data.csv', index=False)

print("\n✅ Saved all Model 1 files successfully:")
print("- models/model1_binary.pkl")
print("- models/scaler.pkl") 
print("- models/encoders.pkl")
print("- models/feature_selector.pkl")
print("- final_model_data.csv")

# Save feature information for consistency
feature_info = {
    'all_features': list(X.columns),
    'selected_features': list(selected_features),
    'categorical_features': categorical_cols,
    'numerical_features': [col for col in X.columns if col not in categorical_cols]
}
save_model_safely(feature_info, 'models/feature_info.pkl')

print("✅ Model 1 training completed successfully!")