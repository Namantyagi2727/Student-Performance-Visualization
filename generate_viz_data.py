"""
ML Visualization Data Generator
This script processes the student performance data and generates JSON files
for interactive visualizations in the web interface.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
import joblib
import os
from sklearn.manifold import TSNE
try:
    import umap.umap_ as umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

# Load dataset
df = pd.read_csv("Student_Performance_Data.csv")

print("📊 Starting data generation for visualizations...")
print(f"Dataset shape: {df.shape}")

# ============================================================================
# 1. BASIC STATISTICS
# ============================================================================
print("\n1️⃣ Generating basic statistics...")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

basic_stats = {
    "total_records": int(df.shape[0]),
    "total_features": int(df.shape[1]),
    "numeric_features": len(numeric_cols),
    "categorical_features": len(categorical_cols),
    "missing_values": int(df.isnull().sum().sum()),
    "target_variable": "semester_gpa",
    "numeric_features_list": numeric_cols,
    "categorical_features_list": categorical_cols
}

with open("data/basic_stats.json", "w") as f:
    json.dump(basic_stats, f, indent=2)

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================
print("2️⃣ Generating correlation analysis...")

corr_matrix = df[numeric_cols].corr().round(3)
gpa_correlations = corr_matrix['semester_gpa'].sort_values(ascending=False)

# Convert to JSON-serializable format
corr_data = {
    "gpa_correlations": gpa_correlations.to_dict(),
    "top_10_positive": gpa_correlations.head(10).to_dict(),
    "top_10_negative": gpa_correlations.tail(10).to_dict(),
    "correlation_matrix": corr_matrix.values.tolist(),
    "correlation_columns": numeric_cols,
    "max_correlation": float(gpa_correlations.iloc[1]),  # Skip self-correlation
    "min_correlation": float(gpa_correlations.iloc[-1])
}

with open("data/correlations.json", "w") as f:
    json.dump(corr_data, f, indent=2)

# ============================================================================
# 3. FEATURE DISTRIBUTIONS
# ============================================================================
print("3️⃣ Generating feature distributions...")

distributions = {}
for col in numeric_cols:
    distributions[col] = {
        "mean": float(df[col].mean()),
        "median": float(df[col].median()),
        "std": float(df[col].std()),
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "q1": float(df[col].quantile(0.25)),
        "q3": float(df[col].quantile(0.75)),
        "skewness": float(df[col].skew()),
        "kurtosis": float(df[col].kurtosis())
    }

with open("data/distributions.json", "w") as f:
    json.dump(distributions, f, indent=2)

# Ensure output directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ============================================================================
# 4. DATA PREPROCESSING FOR ML
# ============================================================================
print("4️⃣ Preprocessing data for ML models...")

# Make a copy to preserve original
df_ml = df.copy()

# Encode categorical columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    le_dict[col] = le

# Target variable
target_column = "semester_gpa"
X = df_ml.drop(target_column, axis=1)
y = df_ml[target_column]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# ============================================================================
# 5. MODEL TRAINING AND COMPARISON
# ============================================================================
print("5️⃣ Training ML models...")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(kernel='rbf'),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

results = {}
model_performance = []

for name, model in models.items():
    print(f"   Training {name}...", end=" ")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    results[name] = {
        "model": model,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "predictions": y_pred_test.tolist()
    }
    
    model_performance.append({
        "name": name,
        "rmse_train": float(rmse_train),
        "rmse_test": float(rmse_test),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "mae_test": float(np.mean(np.abs(y_test - y_pred_test)))
    })
    
    print("✓")

# Save model performance
print("\n🔎 Running cross-validation for model robustness (5-fold)...")
for perf in model_performance:
    name = perf['name']
    clf = results[name]['model']
    try:
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='r2')
        perf['cv_r2_mean'] = float(np.mean(scores))
        perf['cv_r2_std'] = float(np.std(scores))
    except Exception as e:
        perf['cv_r2_mean'] = None
        perf['cv_r2_std'] = None

with open("data/model_performance.json", "w") as f:
    json.dump(model_performance, f, indent=2)

# ==========================================================================
# 11. EMBEDDINGS & POINT-LEVEL DATA FOR FRONT-END
# ==========================================================================
print("11️⃣ Generating embeddings and point-level JSON for front-end...")

# Use scaled features for embeddings (X_scaled corresponds to X.columns)
emb_matrix = X_scaled
embedding_results = {}
if HAVE_UMAP:
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(emb_matrix)
        embedding_results['umap'] = emb.tolist()
        print('   UMAP embedding computed')
    except Exception as e:
        print('   UMAP failed:', e)

# Always compute t-SNE (fallback)
try:
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    emb_tsne = tsne.fit_transform(emb_matrix)
    embedding_results['tsne'] = emb_tsne.tolist()
    print('   t-SNE embedding computed')
except Exception as e:
    print('   t-SNE failed:', e)

# Prepare point-level JSON (sample key features + embeddings)
point_cols = ['semester_gpa'] + numeric_cols
points = []
for i, row in df.iterrows():
    pt = { 'id': int(i) }
    # Add categorical columns of interest
    for c in ['gender', 'country_region']:
        if c in df.columns:
            pt[c] = row[c]
    # Add numeric columns
    for c in numeric_cols:
        try:
            pt[c] = float(row[c]) if not pd.isnull(row[c]) else None
        except Exception:
            pt[c] = None
    # Add embeddings if available
    if 'umap' in embedding_results:
        pt['umap_x'], pt['umap_y'] = embedding_results['umap'][i]
    if 'tsne' in embedding_results:
        pt['tsne_x'], pt['tsne_y'] = embedding_results['tsne'][i]
    points.append(pt)

with open('data/points.json', 'w') as f:
    json.dump({'points': points, 'numeric_features': numeric_cols}, f, indent=2)

# Save embeddings summary
with open('data/embeddings.json', 'w') as f:
    json.dump({'embeddings': embedding_results}, f, indent=2)

# ==========================================================================
# 12. SHAP EXPLANATIONS (if available)
# ==========================================================================
if HAVE_SHAP:
    try:
        print('\n🔬 Computing SHAP values for best model (this may take a few moments)...')
        # Use a TreeExplainer for tree-based models
        if hasattr(best_model, 'predict'):
            try:
                explainer = shap.Explainer(best_model, X_train)
                shap_vals = explainer(X)
                # shap_vals.values shape: (n_samples, n_features)
                mean_abs_shap = np.mean(np.abs(shap_vals.values), axis=0)
                shap_summary = {col: float(val) for col, val in zip(X.columns, mean_abs_shap)}
                with open('data/shap_summary.json', 'w') as f:
                    json.dump({'shap_summary': shap_summary}, f, indent=2)
                # Save small sample of per-sample shap values (first 200)
                sample_shap = shap_vals.values[:200].tolist()
                with open('data/shap_values_sample.json', 'w') as f:
                    json.dump({'feature_names': X.columns.tolist(), 'shap_values_sample': sample_shap}, f, indent=2)
                print('   SHAP summary saved')
            except Exception as e:
                print('   SHAP computation failed:', e)
    except Exception as e:
        print('   SHAP not available or failed:', e)
else:
    print('\nℹ️ SHAP package not installed. To compute SHAP explanations, install `shap` in your virtualenv and re-run this script:')
    print('   python -m pip install shap')

# ============================================================================
# 6. BEST MODEL AND FEATURE IMPORTANCE
# ============================================================================
print("6️⃣ Analyzing best model...")

best_model_name = min(results, key=lambda k: results[k]['rmse_test'])
best_model = results[best_model_name]['model']

print(f"   Best Model: {best_model_name}")

# Feature importance for tree-based models
feature_importance = {}

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    for i, idx in enumerate(indices):
        feature_importance[X.columns[idx]] = float(importances[idx])

# Save best model info
best_model_info = {
    "name": best_model_name,
    "rmse_train": float(results[best_model_name]['rmse_train']),
    "rmse_test": float(results[best_model_name]['rmse_test']),
    "r2_train": float(results[best_model_name]['r2_train']),
    "r2_test": float(results[best_model_name]['r2_test']),
    "feature_importance": feature_importance,
    "feature_count": len(X.columns),
    "training_samples": len(X_train),
    "test_samples": len(X_test)
}

with open("data/best_model_info.json", "w") as f:
    json.dump(best_model_info, f, indent=2)

# ============================================================================
# 7. KEY INSIGHTS
# ============================================================================
print("7️⃣ Generating key insights...")

insights = {
    "gpa_statistics": {
        "mean": float(df['semester_gpa'].mean()),
        "median": float(df['semester_gpa'].median()),
        "std": float(df['semester_gpa'].std()),
        "min": float(df['semester_gpa'].min()),
        "max": float(df['semester_gpa'].max())
    },
    "strongest_positive_factors": list(gpa_correlations.head(6).index)[1:6],  # Skip GPA itself
    "strongest_negative_factors": list(gpa_correlations.tail(5).index),
    "model_insights": {
        "best_model": best_model_name,
        "best_r2": float(results[best_model_name]['r2_test']),
        "improvement_over_baseline": "Feature analysis shows study hours and previous GPA are strongest predictors"
    }
}

with open("data/insights.json", "w") as f:
    json.dump(insights, f, indent=2)

# ============================================================================
# 8. SAMPLE PREDICTIONS
# ============================================================================
print("8️⃣ Generating sample predictions...")

# Create sample student profiles
samples = [
    {
        "name": "High Performer",
        "profile": {
            "study_hours_per_week": 35,
            "prev_sem_gpa": 3.8,
            "attendance_percentage": 95,
            "sleep_hours_per_day": 7,
            "stress_level": 4,
            "english_proficiency_score": 110
        }
    },
    {
        "name": "Average Performer",
        "profile": {
            "study_hours_per_week": 20,
            "prev_sem_gpa": 2.8,
            "attendance_percentage": 75,
            "sleep_hours_per_day": 6,
            "stress_level": 7,
            "english_proficiency_score": 85
        }
    },
    {
        "name": "At Risk",
        "profile": {
            "study_hours_per_week": 8,
            "prev_sem_gpa": 2.0,
            "attendance_percentage": 50,
            "sleep_hours_per_day": 5,
            "stress_level": 9,
            "english_proficiency_score": 70
        }
    }
]

# Save sample profiles for visualization
with open("data/sample_profiles.json", "w") as f:
    json.dump(samples, f, indent=2)

# ============================================================================
# 9. CATEGORICAL FEATURE ANALYSIS
# ============================================================================
print("9️⃣ Analyzing categorical features...")

categorical_analysis = {}
for col in categorical_cols:
    if col in df.columns:
        categorical_analysis[col] = {
            "unique_values": int(df[col].nunique()),
            "value_counts": df[col].value_counts().to_dict(),
            "avg_gpa_by_category": df.groupby(col)['semester_gpa'].mean().to_dict()
        }

with open("data/categorical_analysis.json", "w") as f:
    json.dump(categorical_analysis, f, indent=2)

# ============================================================================
# 10. SAVE MODELS FOR PREDICTION
# ============================================================================
print("🔟 Saving models for server-side predictions...")

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le_dict, "models/encoders.pkl")
joblib.dump(X.columns, "models/feature_names.pkl")

print("\n" + "="*60)
print("✅ ALL DATA GENERATION COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  📁 data/")
print("    ├── basic_stats.json")
print("    ├── correlations.json")
print("    ├── distributions.json")
print("    ├── model_performance.json")
print("    ├── best_model_info.json")
print("    ├── insights.json")
print("    ├── sample_profiles.json")
print("    └── categorical_analysis.json")
print("  📁 models/")
print("    ├── best_model.pkl")
print("    ├── scaler.pkl")
print("    ├── encoders.pkl")
print("    └── feature_names.pkl")
