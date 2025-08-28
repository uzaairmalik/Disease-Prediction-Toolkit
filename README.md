# Disease-Prediction-Toolkit
A beginner-friendly, production-style ML project that predicts the presence of heart disease using the **UCI Heart Disease dataset** (Kaggle).
## 📦 What’s Inside
- Clean, structured **Colab-ready notebook**: data loading, preprocessing, model training, evaluation
- **Three models**: Logistic Regression, Decision Tree, Random Forest
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Visuals**: Histograms, Confusion Matrices, ROC curves, Feature Importance (Random Forest)
- Saved **artifacts**: `heart_rf_model.pkl`, `heart_scaler.pkl`, `heart_user_template.csv`

## 🗂 Dataset
- Kaggle: `redwankarimsony/heart-disease-data` → file `heart_disease_uci.csv`
- Target column: `num` → converted to binary `target` (0 = no disease, 1 = disease)

## 🚀 Quick Start (Colab)
1. Download your `kaggle.json` from Kaggle → Account → API.
2. Upload it in Colab working directory.
3. Open and run `Disease_Prediction_Toolkit.ipynb` cell-by-cell.
4. The notebook will download the dataset, train models, and show evaluation plots.
5. Artifacts are saved at the end for reuse.

## 🧹 Preprocessing
- Missing numeric values → mean
- Missing categorical values → mode
- One-hot encoding for categorical features
- Train/test split with `stratify=y`
- Standardization using `StandardScaler`

## 🧠 Models
- Logistic Regression (`max_iter=1000`)
- Decision Tree (default params, `random_state=42`)
- Random Forest (`n_estimators=200`, `random_state=42`)

## 📊 Evaluation
- `accuracy_score`, `classification_report`
- **Confusion Matrix** for each model
- **ROC Curves** + **AUC** comparison
- **Feature Importance** (Random Forest)

## 💾 Artifacts
- `heart_rf_model.pkl` – trained RF model
- `heart_scaler.pkl` – fitted scaler
- `heart_user_template.csv` – correct feature headers for batch prediction
## 🌱 Future Work
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Cross-validation (StratifiedKFold)
- Class imbalance strategies (class_weight/SMOTE)
- Add more models (XGBoost, SVM), model stacking
- SHAP for explainability

## 🙌 Credits
- Dataset: UCI Heart Disease (Kaggle mirror)
- Built with: pandas, scikit-learn, matplotlib
