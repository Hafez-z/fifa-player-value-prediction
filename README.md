# FIFA Player Scouting System — Full ML Pipeline

## Overview
A complete machine learning project across two assignments on the FIFA dataset
(~19,000 players). The goal: predict a player's **market value (M$)** and
classify their **performance tier** (Low / Mid / High / Elite).

## What's Inside

### Assignment 2 — Regression & Classification Foundations
- **EDA**: distributions, correlation heatmap, skewness analysis, outlier detection
- **Preprocessing**: Winsorization, log transformation, StandardScaler,
  OneHotEncoder, TargetEncoder (no data leakage)
- **Regression**: Simple Linear, Multiple Linear, Polynomial (degrees 1–4),
  Ridge (L2), Lasso (L1) with alpha sweep and cross-validation
- **Classification**: Logistic Regression (L1 & L2), GaussianNB,
  BernoulliNB, ComplementNB
- **Model Evaluation**: K-Fold and Stratified K-Fold cross-validation,
  confusion matrices, learning curves

### Assignment 3 — Advanced Models & Ensembles
- **Models**: KNN, SVM (RBF kernel), Decision Tree — both classifier and regressor
  versions with GridSearchCV / RandomizedSearchCV tuning
- **Ensembles**: Random Forest (200 trees), Hard Voting, Soft Voting,
  Voting Regressor, Stacking (KNN + SVM + DT + RF → meta LogReg/Ridge)
- **Unified Scouting System**: single pipeline that takes raw player data
  and returns both market value and performance tier simultaneously
- **Stability Assessment**: 5-fold CV with mean ± std for all ensemble methods
- **Visualizations**: residual plots, violin CV distributions,
  full model comparison bar charts, validation curves

## Tools & Libraries
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · SciPy

## Results
| Task | A2 Baseline | A3 Advanced (Stacking) |
|------|:-----------:|:----------------------:|
| Regression R² | 0.78 | ~0.95+ |
| Classification Accuracy | 81% | ~91%+ |
