# spamdetection
# Requirements:
# pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn xgboost

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: SMOTE for oversampling
try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except Exception:
    smote_available = False

# Optional: XGBoost
try:
    import xgboost as xgb
    xgb_available = True
except Exception:
    xgb_available = False

# ========== 1) Create synthetic imbalanced dataset ==========
# For real data, replace this with: df = pd.read_csv('creditcard.csv')
X, y = make_classification(
    n_samples=100000, n_features=30, n_informative=10, n_redundant=10,
    n_clusters_per_class=2, weights=[0.995, 0.005], flip_y=0.01, random_state=42
)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df['target'] = y

print("Dataset shape:", df.shape)
print("Positive (fraud) ratio:", df['target'].mean())

# ========== 2) Train/test split (stratified) ==========
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== 3) Preprocessing: scaling ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4) Handle imbalance: SMOTE (optional) ==========
if smote_available:
    print("SMOTE available: performing oversampling on training set...")
    sm = SMOTE(random_state=42)   # ✅ no n_jobs here
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    print("After SMOTE pos ratio:", y_train_res.mean())
else:
    print("SMOTE not available: using original imbalanced training set.")
    X_train_res, y_train_res = X_train_scaled, y_train

# ========== 5) Train baseline model: Random Forest ==========
rf = RandomForestClassifier(
    n_estimators=200, class_weight='balanced_subsample', n_jobs=-1, random_state=42
)
rf.fit(X_train_res, y_train_res)

# Predictions & probabilities
y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

# ========== 6) Evaluation helper ==========
def evaluate_model(y_true, y_pred, y_proba, model_name="model"):
    print(f"\n--- Evaluation: {model_name} ---")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    roc = roc_auc_score(y_true, y_proba)
    print(f"ROC AUC: {roc:.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")

    return roc, pr_auc, precision, recall

rf_results = evaluate_model(y_test, y_pred, y_proba, model_name="RandomForest (baseline)")

# ========== 7) Optional: train XGBoost (if installed) ==========
if xgb_available:
    print("\nTraining XGBoost...")
    xgclf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, scale_pos_weight=(1.0 / y_train.mean())-1.0,
        use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1
    )
    xgclf.fit(X_train_res, y_train_res)
    y_pred_xgb = xgclf.predict(X_test_scaled)
    y_proba_xgb = xgclf.predict_proba(X_test_scaled)[:, 1]
    xgb_results = evaluate_model(y_test, y_pred_xgb, y_proba_xgb, model_name="XGBoost")

# ========== 8) Threshold tuning (precision/recall tradeoff) ==========
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# pick threshold for target recall or precision; example: tune for recall >= 0.80
target_recall = 0.8
idx = np.where(recall >= target_recall)[0]
if idx.size:
    chosen_idx = idx[-1]  # last index with recall >= target_recall
    chosen_threshold = thresholds[chosen_idx] if chosen_idx < thresholds.shape[0] else thresholds[-1]
    print(f"\nChosen threshold for recall >= {target_recall}: {chosen_threshold:.4f}")
else:
    chosen_threshold = 0.5
    print("\nCould not reach target recall with available thresholds; defaulting to 0.5")

# Apply threshold
y_pred_thresh = (y_proba >= chosen_threshold).astype(int)
print("\nEvaluation at chosen threshold:")
print(classification_report(y_test, y_pred_thresh, digits=4))

# ========== 9) Plot PR curve ==========
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve (RandomForest)')
plt.grid(True)
plt.legend()
plt.show()

# ========== 10) Quick feature importance (RF) ==========
importances = rf.feature_importances_
fi = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
print("\nTop 10 feature importances (RandomForest):")
print(fi)

sns.barplot(x=fi.values, y=fi.index)
plt.title("Top 10 Feature Importances (RF)")
plt.tight_layout()
plt.show()
