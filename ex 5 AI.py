
import pandas as pd

# Step 1: Load the dataset
file_path = "bank (1).csv"
data = pd.read_csv(file_path, delimiter=';')

print("Dataset Info:")
print(data.info())

print("\nFirst Few Rows of Data:")
print(data.head())


selected_columns = ['y', 'job', 'marital', 'default', 'housing', 'poutcome']  # 'poutcome' as per dataset
df2 = data[selected_columns]


print("\nExtracted DataFrame (df2):")
print(df2.head())

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

print("\nTransformed DataFrame (df3):")
print(df3.head())


df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

print("\nTransformed DataFrame (df3):")
print(df3.head())

import seaborn as sns
import matplotlib.pyplot as plt

df3['y'] = df3['y'].apply(lambda val: 1 if val == 'yes' else 0)
correlation_matrix = df3.corr()


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Heatmap of Correlation Coefficients")
plt.show()

y = df3['y']
X = df3.drop(columns=['y'])

print("\nTarget Variable (y):")
print(y.head())

print("\nExplanatory Variables (X):")
print(X.head())


correlation_matrix = df3.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Heatmap of Correlation Coefficients")
plt.show()

y = df3['y']
X = df3.drop(columns=['y'])

print("\nTarget Variable (y):")
print(y.head())

print("\nExplanatory Variables (X):")
print(X.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

logistic_conf_matrix = confusion_matrix(y_test, y_pred_logistic)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

plt.figure(figsize=(6, 4))
sns.heatmap(logistic_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

knn_conf_matrix = confusion_matrix(y_test, y_pred_knn)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

plt.figure(figsize=(6, 4))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - K-Nearest Neighbors (k=3)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(f"K-Nearest Neighbors (k=3) Accuracy: {knn_accuracy:.2f}")

print("\nComparison of Models:")
print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
print(f"K-Nearest Neighbors (k=3) Accuracy: {knn_accuracy:.2f}")

