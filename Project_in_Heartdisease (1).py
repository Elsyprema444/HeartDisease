#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:14:36 2024

@author: elsyp
"""

import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Importing Data's
cancer_data=pd.read_csv(r"C:\Users\elsyp\OneDrive\Desktop\Downloads\heart+disease\processed.cleveland.data")




# Assign meaningful column labels
cancer_data.columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
              'rest_ecg', 'max_heart_rate', 'exercise_angina', 'st_depression', 'slope', 'num_major_vessels',
              'thalassemia', 'target']


# Descriptive Statistics
print(cancer_data.describe())
# Histogram
sns.histplot(cancer_data['target'], kde=True)
plt.show()


# Display the DataFrame with your own column labels



# Separating only the specific columns:
fifth_columns = cancer_data.iloc[:,6]

# Finding missing values:
missing_values = cancer_data.isnull().sum()

# Finding missing values in the specific rows
cancer_data.replace('0', pd.NA, inplace=True)

missing_values_row6 = cancer_data.iloc[:,6].isnull().sum()
print(f"missing_values_row6: {missing_values_row6}")


#So, we found, there is no missing values! the data set is perfect as well as `0'represents valid value.
#Now, we have to consider, from the data set, which column is consider to relavent to predict the heart disease.
#For that we can use two techniques.1.correlation analysis 2.Domain Knowledge

# Before normalization, we need to ensure all values are numeric where possible
# Convert all columns to numeric, coercing errors to NaN (non-numeric strings become NaN)
cancer_data_numeric = cancer_data.apply(pd.to_numeric, errors='coerce')

# Now we handle NaN values. For this example, we'll fill NaN with the column mean.
# This is just an example strategy. The strategy for handling NaNs should be chosen based on the dataset and analysis requirements.
cancer_data_filled = cancer_data_numeric.fillna(cancer_data_numeric.mean())

# Now, we can safely apply z-score standardization
standardization = cancer_data_numeric.apply(zscore)

## Untill hier, we have done with preprocessed the data.

#2. Data Processing:
#correlation analysis is to predict which column is to predict the heart disease: 
    # continious variable:-1 to 1(numerical, with numbers, blood pressure, age, cholestral)
    # categorical variable: 0 and 1(yes or no, male or female)
    #1. correlation matrix:
        # many steps:1. pearsons correlation coefficient, 2.Chi-square test, 3.ANOVA, 4 Cramers.V



#For Continuous Variables: High absolute values of Pearson's correlation coefficient 
#indicate a strong relationship with the target variable.

# For Data Processing, we have selected correlation matrix to find out which 
#column is relevant to predict the cancer disease.        
correlation_matrix = cancer_data_filled.corr()
correlation_matrix = cancer_data_filled.corr()
print(correlation_matrix['target'])

sns.histplot(correlation_matrix['target'], kde=True)
plt.show()
#For Categorical Variables: A low p-value in the Chi-square test suggests a significant 
# association with the target variable.
# Example for a categorical variable 'sex' and 'target'
contingency_table = pd.crosstab(cancer_data_filled['sex'], cancer_data_filled['target'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2 Statistic: {chi2}, p-value: {p}")

# 3. Model Developement
# Regression model for continious variable and classification model is for categorical variable
# For Continious Variable: Linear regression and lasso regresssion
# For Categrical Variable: Logistic regression, decision tree, random forest,gradient boosting machine.




# Assuming 'target' is your continuous outcome and you've selected some features based on correlation
X = cancer_data_filled[['age', 'resting_blood_pressure', 'cholesterol']]  # example features
y = cancer_data_filled['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% testing
# For a simple linear regression


# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Compute and print the metrics
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Coefficient of Determination (R^2):", r2_score(y_test, y_pred))

#MSE gives you the average squared difference between 
#the estimated values and the actual value. A lower MSE indicates 
#a better fit to the data.

#R^2 measures the proportion of the variance in 
# the dependent variable that is predictable from the independent variables.
# R^2 values range from 0 to 1, where higher values indicate a better fit.


# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import your data
cancer_data = pd.read_csv(r"C:\Users\elsyp\OneDrive\Desktop\Downloads\heart+disease\processed.cleveland.data")

# Define column names
cancer_data.columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                       'rest_ecg', 'max_heart_rate', 'exercise_angina', 'st_depression', 'slope', 'num_major_vessels',
                       'thalassemia', 'target']

# Convert target to binary if it's not already
cancer_data['target'] = (cancer_data['target'] != 0).astype(int)

# Prepare data for training
X = cancer_data[['sex', 'resting_blood_pressure', 'cholesterol']]  # You can choose other features too
y = cancer_data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[4]:


from sklearn.metrics import roc_curve, auc

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[5]:


from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
average_precision = average_precision_score(y_test, y_pred)

# Plot Precision-Recall curve
plt.figure()
plt.step(recall, precision, where='post', label='Precision-Recall curve (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()


# In[6]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import RandomForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import your data
cancer_data = pd.read_csv(r"C:\Users\elsyp\OneDrive\Desktop\Downloads\heart+disease\processed.cleveland.data")

# Define column names
cancer_data.columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                       'rest_ecg', 'max_heart_rate', 'exercise_angina', 'st_depression', 'slope', 'num_major_vessels',
                       'thalassemia', 'target']

# Convert target to binary if it's not already
cancer_data['target'] = (cancer_data['target'] != 0).astype(int)

# Prepare data for training
X = cancer_data[['sex', 'resting_blood_pressure', 'cholesterol']]  # Consider selecting more features
y = cancer_data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[8]:


from sklearn.metrics import roc_curve, auc

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[9]:


from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
average_precision = average_precision_score(y_test, y_pred)

# Plot Precision-Recall curve
plt.figure()
plt.step(recall, precision, where='post', label='Precision-Recall curve (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()


# In[10]:


# Hyperparameter tuning with grid search
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the model
rf = RandomForestClassifier(random_state=42)

# Set up the parameters grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# In[11]:


#2. Feature engineering:
# Example of creating a new feature by combining existing features
cancer_data['combined_feature'] = cancer_data['age'] * cancer_data['cholesterol']

# Include this new feature in your model training
X = cancer_data[['sex', 'resting_blood_pressure', 'cholesterol', 'combined_feature']]  # Adjust as needed
y = cancer_data['target']


# In[12]:


#3. cross validation:
from sklearn.model_selection import cross_val_score

# Using the best parameters from the GridSearch
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)

# Perform cross-validation
scores = cross_val_score(best_rf, X, y, cv=5)  # 5-fold cross-validation
print("Cross-validated scores:", scores)
print("Average score:", scores.mean())


# In[13]:


#4. Handling imbalanced data
from sklearn.model_selection import cross_val_score

# Using the best parameters from the GridSearch
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)

# Perform cross-validation
scores = cross_val_score(best_rf, X, y, cv=5)  # 5-fold cross-validation
print("Cross-validated scores:", scores)
print("Average score:", scores.mean())


# In[14]:





# In[15]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Assuming you already have y_test and y_pred from your model predictions
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nFull Classification Report:\n", classification_report(y_test, y_pred))


# In[16]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[18]:


from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# classification metrics:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average='binary'))
print("Recall:", recall_score(y_test, y_pred_dt, average='binary'))
print("F1 Score:", f1_score(y_test, y_pred_dt, average='binary'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

#ROC curve and AUC
from sklearn.metrics import roc_curve, auc

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, dt_model.predict_proba(X_test)[:,1])
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.figure()
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Decision Tree)')
plt.legend(loc="lower right")
plt.show()
# confusion amtrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt="d", cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Decision Tree')
plt.show()


# In[ ]:




