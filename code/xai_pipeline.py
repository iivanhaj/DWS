import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import exposure
from skimage.measure import shannon_entropy
from skimage.util import img_as_ubyte

# Handle different skimage versions for GLCM
try:
    from skimage.feature import greycomatrix, greycoprops
except ImportError:
    try:
        from skimage.feature.texture import greycomatrix, greycoprops
    except ImportError:
        # If both fail, we'll define dummy functions
        greycomatrix = None
        greycoprops = None

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import shap
import joblib


DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
DATASET_DIR = os.path.abspath(DATASET_DIR)
METRICS_CSV = os.path.join(DATASET_DIR, 'metrics_with_ber.csv')
FEATURES_CSV = os.path.join(DATASET_DIR, 'features.csv')
MASTER_CSV = os.path.join(DATASET_DIR, 'master_ml_dataset.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def extract_features_from_path(img_path):
    """Extract a small, robust set of features from an image file path.
    Returns a dict of features.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = rgb2gray(img_rgb)
        gray_u8 = img_as_ubyte(gray)

        features = {}
        # simple intensity stats
        features['mean_intensity'] = float(np.mean(gray_u8))
        features['std_intensity'] = float(np.std(gray_u8))

        # entropy
        try:
            features['entropy'] = float(shannon_entropy(gray_u8))
        except Exception:
            features['entropy'] = float(0.0)

        # edge density (Canny)
        try:
            edges = canny(gray, sigma=1.0)
            features['edge_density'] = float(np.sum(edges) / edges.size)
        except Exception:
            features['edge_density'] = float(0.0)

        # simple contrast via RMS contrast (approx)
        try:
            p2, p98 = np.percentile(gray_u8, (2, 98))
            features['contrast'] = float((p98 - p2) / 255.0)
        except Exception:
            features['contrast'] = float(0.0)

        # GLCM contrast as additional texture feature (on a small patch)
        features['glcm_contrast'] = float(0.0)
        if greycomatrix is not None and greycoprops is not None:
            try:
                # downsample to speed up
                small = cv2.resize(gray_u8, (128, 128), interpolation=cv2.INTER_AREA)
                # quantize to 8 levels to keep GLCM small
                bins = np.linspace(0, 256, 9)
                quant = np.digitize(small, bins) - 1
                glcm = greycomatrix(quant, distances=[1], angles=[0], levels=9, symmetric=True, normed=True)
                glcm_contrast = greycoprops(glcm, prop='contrast')[0, 0]
                features['glcm_contrast'] = float(glcm_contrast)
            except Exception:
                features['glcm_contrast'] = float(0.0)

        # histogram features
        try:
            hist = exposure.histogram(gray_u8)[0].astype(float)
            hist = hist / (hist.sum() + 1e-9)
            # first three histogram moments
            features['hist_entropy'] = float(-np.sum(hist * np.log(hist + 1e-9)))
            features['hist_skew'] = float(((np.arange(len(hist)) - np.mean(hist))**3 * hist).sum())
        except Exception:
            features['hist_entropy'] = 0.0
            features['hist_skew'] = 0.0

        return features
    except Exception as exc:
        print(f"Feature extraction failed for {img_path}: {exc}")
        return None


def build_features_dataframe(metrics_df, overwrite=False):
    """Build or load features DataFrame. Returns DataFrame with columns ['image','method','attack','split', <features>]."""
    if os.path.exists(FEATURES_CSV) and not overwrite:
        print(f"Loading existing features from {FEATURES_CSV}")
        return pd.read_csv(FEATURES_CSV)

    rows = []
    missing = 0
    total = len(metrics_df)
    print(f"Extracting features for {total} rows (this may take a while)...")
    for _, r in tqdm(metrics_df.iterrows(), total=total):
        img_name = r['image']
        method = r['method']
        attack = r['attack']
        split = r.get('split', 'train')

        # Construct expected attacked image path
        attack_img = os.path.join(DATASET_DIR, 'attacks', method, attack, split, img_name)
        if not os.path.exists(attack_img):
            # fallback: maybe different capitalization or location
            attack_img = os.path.join(DATASET_DIR, 'attacks', method, attack, split, img_name)

        features = None
        if os.path.exists(attack_img):
            features = extract_features_from_path(attack_img)
        else:
            missing += 1

        if features is None:
            # add NaNs so merging keeps index aligned
            features = {
                'mean_intensity': np.nan,
                'std_intensity': np.nan,
                'entropy': np.nan,
                'edge_density': np.nan,
                'contrast': np.nan,
                'glcm_contrast': np.nan,
                'hist_entropy': np.nan,
                'hist_skew': np.nan,
            }

        row = {'image': img_name, 'method': method, 'attack': attack, 'split': split}
        row.update(features)
        rows.append(row)

    print(f"Missing images: {missing} / {total}")
    features_df = pd.DataFrame(rows)
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved features to {FEATURES_CSV}")
    return features_df


def train_and_explain(master_df):
    # Ensure consistent column names
    df = master_df.copy()
    if 'image' in df.columns:
        df = df.rename(columns={'image': 'Image_ID'})

    # Drop rows with missing features or BER
    df = df.dropna(subset=['BER'])
    feature_cols = ['mean_intensity', 'std_intensity', 'entropy', 'edge_density', 'contrast', 'glcm_contrast', 'hist_entropy', 'hist_skew']
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols + ['method', 'attack']].copy()
    X = pd.get_dummies(X, columns=['method', 'attack'])
    y = df['BER'].astype(float)

    # Keep feature names for plotting
    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split via indices so we can keep DataFrame for SHAP
    idx = np.arange(X_scaled.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")

    # Save metrics
    with open(os.path.join(RESULTS_DIR, 'xai_metrics.txt'), 'w') as fh:
        fh.write(f"MAE: {mae}\nR2: {r2}\n")

    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_DIR, 'rf_ber_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print(f"Saved model and scaler to {MODELS_DIR}")

    # SHAP explainability
    print("Computing SHAP values (this may take a while)...")
    explainer = shap.TreeExplainer(model)
    # Use the DataFrame representation for SHAP plots
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, show=False)
    plt.tight_layout()
    out_summary = os.path.join(RESULTS_DIR, 'shap_summary.png')
    plt.savefig(out_summary, dpi=300)
    plt.close()
    print(f"Saved SHAP summary to {out_summary}")

    # Feature importances (model-based)
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).head(30)
    plt.figure(figsize=(8, max(4, 0.25 * len(imp_df))))
    sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis')
    plt.title('Top feature importances')
    plt.tight_layout()
    out_imp = os.path.join(RESULTS_DIR, 'feature_importances.png')
    plt.savefig(out_imp, dpi=300)
    plt.close()
    print(f"Saved feature importances to {out_imp}")

    return model, scaler


def main(overwrite_features=False):
    print("Phase 4: XAI Pipeline Started")

    # Step 4.1: Load metrics and extract features
    print("Loading metrics...")
    metrics_df = pd.read_csv(METRICS_CSV)

    features_df = build_features_dataframe(metrics_df, overwrite=overwrite_features)

    # Step 4.2: Merge to create master dataset
    print("Merging features with metrics to build master dataset...")
    master_df = metrics_df.merge(features_df, on=['image', 'method', 'attack', 'split'], how='left')
    master_df.to_csv(MASTER_CSV, index=False)
    print(f"Saved master dataset to {MASTER_CSV}")

    # Step 4.3: Train model and evaluate
    print("Training model and computing explanations...")
    model, scaler = train_and_explain(master_df)

    print("XAI Pipeline Complete. Results and models saved.")


if __name__ == '__main__':
    main()