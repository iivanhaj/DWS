import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from sklearn.model_selection import train_test_split

# Directories
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
DATASET_DIR = os.path.abspath(DATASET_DIR)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("Resuming SHAP analysis from saved model...")
    
    # Load the saved model, scaler, and data
    print("Loading model and data...")
    model = joblib.load(os.path.join(MODELS_DIR, 'rf_ber_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    master_df = pd.read_csv(os.path.join(DATASET_DIR, 'master_ml_dataset.csv'))
    
    print(f"Loaded master dataset with {len(master_df)} rows")
    
    # Recreate the exact same test set using the same random state
    feature_cols = ['mean_intensity', 'std_intensity', 'entropy', 'edge_density', 
                   'contrast', 'glcm_contrast', 'hist_entropy', 'hist_skew']
    
    # Clean data (same as original)
    df = master_df.dropna(subset=feature_cols + ['BER'])
    print(f"After cleaning: {len(df)} rows")
    
    X = df[feature_cols + ['method', 'attack']].copy()
    X = pd.get_dummies(X, columns=['method', 'attack'])
    y = df['BER'].astype(float)
    
    feature_names = X.columns.tolist()
    print(f"Number of features: {len(feature_names)}")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Recreate the exact same train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Test set size: {len(X_test)} samples")
    
    # SHAP on subset (500 samples for reasonable time)
    shap_sample_size = 500
    print(f"Computing SHAP on {shap_sample_size} samples (this will take 10-30 minutes)...")
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_subset = X_test_df.iloc[:shap_sample_size]
    y_test_subset = y_test.iloc[:shap_sample_size]
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_subset)
    
    print("SHAP computation complete! Creating visualizations...")
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_subset, feature_names=feature_names, show=False)
    plt.tight_layout()
    shap_summary_path = os.path.join(RESULTS_DIR, 'shap_summary_subset.png')
    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary to {shap_summary_path}")
    
    # 2. SHAP Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_subset, feature_names=feature_names, 
                     plot_type="bar", show=False)
    plt.tight_layout()
    shap_bar_path = os.path.join(RESULTS_DIR, 'shap_feature_importance.png')
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP feature importance to {shap_bar_path}")
    
    # 3. Traditional Feature Importance (for comparison)
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    rf_imp_path = os.path.join(RESULTS_DIR, 'rf_feature_importances.png')
    plt.savefig(rf_imp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved RF feature importances to {rf_imp_path}")
    
    # 4. Save SHAP values for further analysis
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{name}" for name in feature_names])
    shap_df['predicted_BER'] = model.predict(X_test_subset)
    shap_df['actual_BER'] = y_test_subset.values
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_values_subset.csv'), index=False)
    print(f"Saved SHAP values to {os.path.join(RESULTS_DIR, 'shap_values_subset.csv')}")
    
    # Print top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\nTop 10 Most Important Features (by mean |SHAP|):")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    print("\nSHAP analysis complete! Check the 'results' directory for outputs.")

if __name__ == '__main__':
    main()