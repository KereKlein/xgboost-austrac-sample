
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load and preprocess data
df = pd.read_csv("Data/bank-additional-full.csv", sep=';')
for col in df.select_dtypes(include='object').columns:
    if col != 'y':
        df[col] = LabelEncoder().fit_transform(df[col])
df['y'] = df['y'].map({'yes': 1, 'no': 0})
X = df.drop('y', axis=1)
y = df['y']

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

# Define Optuna optimization
def objective(trial):
    model = XGBClassifier(
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.3, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='auc'
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
best_trial = study.best_trial
print("Best trial parameters:", best_trial.params)
print("Best AUC:", best_trial.value)

# Manually set best parameters (safe subset)
final_model = XGBClassifier(
    learning_rate=best_trial.params['learning_rate'],
    max_depth=best_trial.params['max_depth'],
    min_child_weight=best_trial.params['min_child_weight'],
    subsample=best_trial.params['subsample'],
    colsample_bytree=best_trial.params['colsample_bytree'],
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='auc'
)

# Fit final model
X_train_val_full = pd.concat([X_train, X_val])
y_train_val_full = pd.concat([y_train, y_val])
final_model.fit(X_train_val_full, y_train_val_full)

# AUC scores
train_val_preds = final_model.predict_proba(X_train_val_full)[:, 1]
test_preds = final_model.predict_proba(X_test)[:, 1]
train_val_auc = roc_auc_score(y_train_val_full, train_val_preds)
test_auc = roc_auc_score(y_test, test_preds)

# ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train_val_full, train_val_preds)
fpr_test, tpr_test, _ = roc_curve(y_test, test_preds)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Train+Val AUC = {train_val_auc:.4f}")
plt.plot(fpr_test, tpr_test, label=f"Test AUC = {test_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve with AUC Scores")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Output/auc_plot.png")

# Feature importance
xgb.plot_importance(final_model)
plt.tight_layout()
plt.savefig("Output/feature_importance.png")

# Manual Partial Dependence Plot for Top Features
top_features = list(X.columns[:6])
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

for i, feature in enumerate(top_features):
    ax = axs[i // 3, i % 3]
    grid = np.linspace(X_train_val_full[feature].min(), X_train_val_full[feature].max(), 20)
    avg_preds = []
    for val in grid:
        X_temp = X_train_val_full.copy()
        X_temp[feature] = val
        preds = final_model.predict_proba(X_temp)[:, 1]
        avg_preds.append(np.mean(preds))
    ax.plot(grid, avg_preds)
    ax.set_title(f"PDP for {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Avg. Predicted Probability")
    ax.grid()

plt.tight_layout()
plt.savefig("Output/partial_dependence_plots.png")

# Save model
joblib.dump(final_model, "Output/final_model_xgb.joblib")
