
**Heart Disease Analysis Project**
This repository contains a comprehensive analysis of heart disease data, utilizing various machine learning techniques to predict the presence of heart disease. The project was created using Python and Jupyter Notebook.
**
Project Overview**
This project involves data preprocessing, exploratory data analysis, and the application of machine learning models to predict heart disease. The data used in this project is the processed Cleveland heart disease dataset.


**Files in the Repository**
Project_in_Heartdisease-Copy1.pdf: The detailed report of the project in PDF format.
Project_in_Heartdisease.ipynb: The Jupyter Notebook containing the code and analysis (not uploaded yet, please add this if you have the file).
Data Source
The dataset used in this project is sourced from the UCI Machine Learning Repository:

**Heart Disease Data Set**
**Project Structure**

**Data Preprocessing:**

Load and clean the dataset.
Handle missing values and convert categorical variables.
Standardize numerical variables using z-score normalization.
**Exploratory Data Analysis (EDA):**

Descriptive statistics and visualizations to understand the data.
Correlation analysis to identify relevant features.
**Model Development:**

Implement various machine learning models, including Linear Regression, Logistic Regression, Decision Trees, and Random Forest.
Split the dataset into training and testing sets.
Train the models and evaluate their performance using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.
**Model Evaluation:**

Confusion Matrix to visualize the performance of classification models.
ROC Curve and Precision-Recall Curve to evaluate the models.
Requirements
**To run this project, you need the following libraries:**

pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
**Model Development and Analysis**
**1.Linear Regression**
**Purpose**: To predict continuous outcomes based on independent variables.

**Features Used:** age, resting_blood_pressure, cholesterol
**Steps:**
Split the data into training and testing sets.
Train the model on the training data.
Evaluate the model using Mean Squared Error (MSE) and R-squared (R²) on the test data.
**Results:**
Mean Squared Error (MSE): 1.41
Coefficient of Determination (R²): 0.02
**2.Logistic Regression**
**Purpose: **To predict binary outcomes.

Features Used: sex, resting_blood_pressure, cholesterol
**Steps:**
Convert the target variable to binary.
Split the data into training and testing sets.
Train the model on the training data.
Evaluate the model using accuracy, precision, recall, F1 score, and confusion matrix on the test data.
**Results:**
Accuracy: 62.64%
Precision: 0.62
Recall: 0.53
F1 Score: 0.57
Confusion Matrix:


[[34, 14],
 [20, 23]]
**3.Decision Tree Classifier**
**Purpose: **To predict outcomes by learning simple decision rules inferred from the data features.
**
Features Used:** sex, resting_blood_pressure, cholesterol
**Steps:**
Split the data into training and testing sets.
Train the model on the training data.
Evaluate the model using accuracy, precision, recall, F1 score, and confusion matrix on the test data.
**Results:**
Accuracy: 60.44%
Precision: 0.59
Recall: 0.56
F1 Score: 0.57
Confusion Matrix:


[[32, 16],
 [22, 21]]
**4. Random Forest Classifier**
**Purpose:** To improve the performance by averaging the results of multiple decision trees.

**Features Used:** sex, resting_blood_pressure, cholesterol
**Steps:**
Split the data into training and testing sets.
Train the model on the training data.
Evaluate the model using accuracy, precision, recall, F1 score, and confusion matrix on the test data.
Results:
Accuracy: 58.24%
Precision: 0.57
Recall: 0.49
F1 Score: 0.53
Confusion Matrix:


[[32, 16],
 [22, 21]]
Model Evaluation Metrics
Confusion Matrix:

Visual representation of true positives, true negatives, false positives, and false negatives.
Helps in understanding the performance of classification models.
ROC Curve:

Plots true positive rate against false positive rate.
Area Under the Curve (AUC) provides a single measure of overall model performance.
Precision-Recall Curve:

Precision: The ratio of correctly predicted positive observations to the total predicted positives.
Recall: The ratio of correctly predicted positive observations to the all observations in actual class.
Helps in understanding the trade-off between precision and recall for different thresholds.
Visualizations
Confusion Matrix Heatmap: Used to visualize the performance of classification models.
ROC Curve: Used to evaluate the performance of classification models by plotting the true positive rate against the false positive rate.
Precision-Recall Curve: Used to understand the trade-off between precision and recall.
Conclusion
Each model has its strengths and weaknesses. In this project, the Random Forest and Decision Tree models showed promising results but still have room for improvement. Further optimization and fine-tuning, such as hyperparameter tuning and feature engineering, can enhance the performance of these models.

By comparing different models, we can choose the best performing model for predicting heart disease based on the given dataset. This project demonstrates the process of developing, evaluating, and selecting machine learning models for a healthcare-related problem.
