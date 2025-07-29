import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from collections import Counter
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

print("🚀 Starting Model 2 Training (Attack Type Classification)")
print("=" * 60)

# Load the prepared dataset
print("📂 Loading dataset...")
df = pd.read_csv('final_model_data.csv')
print(f"✅ Loaded dataset with shape: {df.shape}")

# Show original distribution
print(f"\n📊 Original label distribution:")
original_counts = df['attack_type'].value_counts()
print(original_counts)

# CRITICAL: Remove ALL normal samples
print(f"\n🔥 Removing normal samples...")
original_size = len(df)
attack_df = df[df['attack_type'] != 'normal'].copy()
removed_count = original_size - len(attack_df)

print(f"✅ Removed {removed_count:,} normal samples")
print(f"✅ Attack-only dataset shape: {attack_df.shape}")

# Double-check normal removal
if 'normal' in attack_df['attack_type'].values:
    print("❌ ERROR: Normal samples still present! Removing again...")
    attack_df = attack_df[attack_df['attack_type'] != 'normal'].copy()

print(f"\n📊 Attack type distribution (after normal removal):")
attack_counts = attack_df['attack_type'].value_counts()
print(attack_counts)
print(f"✅ Unique attack types: {len(attack_counts)}")

# Verify 'normal' is completely gone
unique_attacks = sorted(attack_df['attack_type'].unique())
print(f"✅ Attack classes to train: {unique_attacks}")
assert 'normal' not in unique_attacks, "ERROR: 'normal' still in attack classes!"

# Prepare features
feature_columns = [col for col in attack_df.columns if col not in ['attack_type', 'label', 'binary_label']]
X = attack_df[feature_columns].copy()
y = attack_df['attack_type'].copy()

print(f"\n🔧 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")
print(f"🔧 Feature columns ({len(feature_columns)}): {feature_columns}")

# Load existing preprocessors
print(f"\n📥 Loading existing preprocessors...")
try:
    existing_encoders = joblib.load('models/encoders.pkl')
    existing_scaler = joblib.load('models/scaler.pkl')
    print("✅ Loaded existing encoders and scaler")
except Exception as e:
    print(f"❌ Error loading preprocessors: {e}")
    exit(1)

# Apply categorical encoding
categorical_cols = ['protocol_type', 'service', 'flag']
print(f"\n🔤 Encoding categorical features: {categorical_cols}")

for col in categorical_cols:
    if col in existing_encoders and col in X.columns:
        encoder = existing_encoders[col]
        print(f"  Encoding {col}...")
        
        def safe_encode(value):
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                return encoder.transform([encoder.classes_[0]])[0]
        
        X[col] = X[col].apply(safe_encode)
        print(f"    ✅ {col} encoded successfully")

# Apply scaling
print(f"\n📏 Applying feature scaling...")
X_scaled = existing_scaler.transform(X)
print("✅ Feature scaling completed")

# Smart sampling strategy based on class frequency
print(f"\n⚖️ Applying smart sampling strategy...")

# Categorize classes by frequency
rare_threshold = 50
medium_threshold = 1000

rare_classes = []
medium_classes = []
common_classes = []

for attack_type, count in attack_counts.items():
    if count <= rare_threshold:
        rare_classes.append(attack_type)
    elif count <= medium_threshold:
        medium_classes.append(attack_type)
    else:
        common_classes.append(attack_type)

print(f"📊 Rare classes (<= {rare_threshold}): {len(rare_classes)}")
print(f"📊 Medium classes ({rare_threshold}-{medium_threshold}): {len(medium_classes)}")
print(f"📊 Common classes (> {medium_threshold}): {len(common_classes)}")

# Create stratified sample
sampled_dfs = []

# For rare classes: use all samples
for attack_type in rare_classes:
    subset = attack_df[attack_df['attack_type'] == attack_type]
    sampled_dfs.append(subset)
    print(f"  {attack_type}: {len(subset)} samples (all)")

# For medium classes: use all or cap at 800
for attack_type in medium_classes:
    subset = attack_df[attack_df['attack_type'] == attack_type]
    if len(subset) > 800:
        subset = subset.sample(n=800, random_state=42)
    sampled_dfs.append(subset)
    print(f"  {attack_type}: {len(subset)} samples")

# For common classes: sample down significantly
for attack_type in common_classes:
    subset = attack_df[attack_df['attack_type'] == attack_type]
    # Sample proportionally but cap
    sample_size = min(2000, max(500, len(subset) // 50))
    subset = subset.sample(n=sample_size, random_state=42)
    sampled_dfs.append(subset)
    print(f"  {attack_type}: {len(subset)} samples (sampled)")

# Combine all samples
balanced_df = pd.concat(sampled_dfs, ignore_index=True)
balanced_df = shuffle(balanced_df, random_state=42)

print(f"\n✅ Final dataset shape: {balanced_df.shape}")
print("Final class distribution:")
final_counts = balanced_df['attack_type'].value_counts()
print(final_counts)

# Prepare final features
X_final = balanced_df[feature_columns].copy()
y_final = balanced_df['attack_type'].copy()

# Apply preprocessing to final data
for col in categorical_cols:
    if col in existing_encoders and col in X_final.columns:
        encoder = existing_encoders[col]
        X_final[col] = X_final[col].apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ 
            else encoder.transform([encoder.classes_[0]])[0]
        )

X_final_scaled = existing_scaler.transform(X_final)

# Create train-test split
print(f"\n🔄 Creating train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final_scaled, y_final, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_final
)

print(f"✅ Training set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")
print(f"✅ Training classes: {len(y_train.unique())}")

# Define optimized models
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced_subsample',  # Better for imbalanced data
        n_jobs=-1,
        bootstrap=True
    ),
    
    'DecisionTree': DecisionTreeClassifier(
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    )
}

# Train and evaluate models
print(f"\n🤖 Training models...")
results = {}

for name, model in models.items():
    print(f"\n  🔥 Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"    ✅ {name} Accuracy: {accuracy:.4f}")
    print(f"    📊 {name} Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"    📊 {name} Macro F1: {report['macro avg']['f1-score']:.4f}")
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'weighted_f1': report['weighted avg']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'report': report
    }

# Select best model (prioritize macro F1 for balanced performance across all classes)
best_name = max(results, key=lambda k: results[k]['macro_f1'])
best_model = results[best_name]['model']
best_accuracy = results[best_name]['accuracy']
best_macro_f1 = results[best_name]['macro_f1']
best_weighted_f1 = results[best_name]['weighted_f1']

print(f"\n🏆 Best Model: {best_name}")
print(f"📊 Accuracy: {best_accuracy:.4f}")
print(f"📊 Macro F1: {best_macro_f1:.4f}")
print(f"📊 Weighted F1: {best_weighted_f1:.4f}")

# Verify model classes
print(f"\n🔍 Model validation...")
print(f"✅ Model classes ({len(best_model.classes_)}): {list(best_model.classes_)}")
assert 'normal' not in best_model.classes_, "ERROR: 'normal' found in model classes!"
print("✅ Verified: No 'normal' class in Model 2")

# Test with Neptune-like sample - CREATE PROPER SAMPLE WITH CORRECT NUMBER OF FEATURES
print(f"\n🧪 Testing with Neptune-like sample...")
print(f"🔧 Model expects {X_train.shape[1]} features")

# Create a sample using the first test sample and modify it to be Neptune-like
if len(X_test) > 0:
    # Use the first test sample as a template
    test_sample = X_test[0:1].copy()  # Keep as 2D array
    
    print(f"✅ Test sample shape: {test_sample.shape}")
    
    try:
        test_proba = best_model.predict_proba(test_sample)[0]
        test_pred = best_model.predict(test_sample)[0]
        
        print(f"🎯 Prediction: {test_pred}")
        print("📊 Top 10 attack probabilities:")
        top_indices = np.argsort(test_proba)[-10:][::-1]
        for i, idx in enumerate(top_indices, 1):
            attack_type = best_model.classes_[idx]
            confidence = test_proba[idx]
            print(f"  {i:2d}. {attack_type:15s}: {confidence:.4f}")
    
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        print("ℹ️  This is just a test - model training was successful")
else:
    print("❌ No test samples available for prediction test")

# Save Model 2
print(f"\n💾 Saving Model 2...")
model_path = 'models/model2_attack_type.pkl'
save_model_safely(best_model, model_path)

# Save comprehensive model metadata
model_info = {
    'model_name': best_name,
    'accuracy': best_accuracy,
    'macro_f1': best_macro_f1,
    'weighted_f1': best_weighted_f1,
    'classes': list(best_model.classes_),
    'num_classes': len(best_model.classes_),
    'feature_names': feature_columns,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'class_distribution': dict(final_counts),
    'training_strategy': 'smart_sampling_by_frequency',
    'training_note': 'Trained ONLY on attack samples - normal completely excluded',
    'no_normal_class': True,
    'rare_classes': rare_classes,
    'medium_classes': medium_classes,
    'common_classes': common_classes,
    'expected_features': X_train.shape[1]  # Add expected feature count
}

info_path = 'models/model2_info.pkl'
save_model_safely(model_info, info_path)

print(f"✅ Model 2 saved to: {model_path}")
print(f"✅ Model info saved to: {info_path}")

# Final detailed report
print(f"\n📋 Detailed Classification Report:")
print("=" * 80)
final_predictions = best_model.predict(X_test)
final_report = classification_report(y_test, final_predictions)
print(final_report)

print(f"\n🎉 Model 2 training completed successfully!")
print(f"✅ Trained on {len(X_train):,} samples")
print(f"✅ {len(best_model.classes_)} attack types can be classified")
print(f"✅ No 'normal' class included")
print(f"✅ Macro F1-score: {best_macro_f1:.4f}")
print(f"✅ Model expects {X_train.shape[1]} features")
print(f"✅ Ready for deployment!")