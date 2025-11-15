# xgboost-austrac-sample
Example XGBoost analysis for EL2 AUSTRACT position

The purpose of the analysis was to demonstrate advanced machine learning capabilities aligned with the EL2 Director, Collaborative Analytics and Data Partnerships role at AUSTRAC. The Python script I developed implements a complete supervised learning workflow using XGBoost, focused on binary classification with the publicly available UCI Bank Marketing dataset.

The workflow includes preprocessing, train-validation-test splitting, and automated hyperparameter tuning using Optuna to optimize model performance (AUC). The final model is trained on the combined training and validation set using the best hyperparameters and evaluated on the test set, with AUC used as the primary metric for model quality.

To enhance interpretability, the analysis includes a feature importance plot and manual Partial Dependence Plots (PDPs) to show how key features influence the predicted probability of the target outcome. These plots enable domain stakeholders to understand model behavior without requiring SHAP or other compatibility-sensitive tools.

The entire workflow is modular and built to reflect production-level model governance and explainability practices.
