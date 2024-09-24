# Fraud-detection
# Import necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('D:\Jupyter notebook\Datasets\creditcard.csv')  

# Display the first few rows to understand the data structure
df.head()

# Check the shape of the dataset (rows, columns)
print(df.shape)

# Check for any missing values
print(df.isnull().sum())

# Get summary statistics of the dataset
print(df.describe())

# Check for the balance of the target variable (Class: 0 - Non-fraud, 1 - Fraud)
print(df['Class'].value_counts())

# Drop the 'Time' column (it may not contribute to the model)
df = df.drop(['Time'], axis=1)

# Features (X) and target (y)
X = df.drop('Class', axis=1)  # All columns except 'Class'
y = df['Class']  # Target variable 'Class'

# Scale the 'Amount' column for better model performance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

from sklearn.model_selection import train_test_split

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict on test data
y_pred_logreg = logreg.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Predict on test data
y_pred_dtree = dtree.predict(X_test)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dtree))
print(classification_report(y_test, y_pred_dtree))

from sklearn.neural_network import MLPClassifier

# Initialize and train the model
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Predict on test data
y_pred_mlp = mlp.predict(X_test)

# Evaluate the model
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train Logistic Regression on the resampled data
logreg.fit(X_train_res, y_train_res)

RESULT

Logistic Regression Accuracy: 0.9990871107053826
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.85      0.57      0.68        98

    accuracy                           1.00     56962
   macro avg       0.92      0.79      0.84     56962
weighted avg       1.00      1.00      1.00     56962

Decision Tree Accuracy: 0.9990519995786665
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.70      0.78      0.74        98

    accuracy                           1.00     56962
   macro avg       0.85      0.89      0.87     56962
weighted avg       1.00      1.00      1.00     56962

Neural Network Accuracy: 0.9994908886626171
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.93      0.77      0.84        98

    accuracy                           1.00     56962
   macro avg       0.96      0.88      0.92     56962
weighted avg       1.00      1.00      1.00     56962

![image](https://github.com/user-attachments/assets/8d4b358f-ad79-4ebe-935d-57460fcb8306)
