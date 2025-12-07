This code is provided to allow to review the implementation of the regression and classification models discussed in the report.

The repository consists of the following scripts corresponding to the methodology detailed in the report:

  - eda.py Performs exploratory data analysis, data cleaning, and feature engineering. This script handles the calculation of RUL and visualizes sensor degradation trends.

  - poly2multi_ridge.ipynb Implements Polynomial Regression with Ridge Regularization. This serves as the baseline non-linear regression model for RUL prediction.

  - gpr_RBF_whitenoise.py Implements Gaussian Process Regression (GPR) using an RBF+white noise kernel. This advanced regression model provides probabilistic predictions with 95% confidence intervals.

  - SVMC_grid_res_RBF.py Implements a Support Vector Machine (SVM) Classifier with an RBF kernel. This script performs binary classification to identify engines approaching failure.
