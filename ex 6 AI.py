
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt



file_path = "data_banknote_authentication.csv"  # Replace with your file path if running locally
data = pd.read_csv("data_banknote_authentication.csv")


print("Dataset Info:")
print(data.info())
print("\nFirst Few Rows of the Data:")
print(data.head())


X = data.drop(columns=['class'])
y = data['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


svc_linear = SVC(kernel='linear', random_state=20)
svc_linear.fit(X_train, y_train)


y_pred_linear = svc_linear.predict(X_test)


conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
print("Classification Report (Linear Kernel):")
print(classification_report(y_test, y_pred_linear))


plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_linear, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Linear Kernel")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


svc_rbf = SVC(kernel='rbf', random_state=20)
svc_rbf.fit(X_train, y_train)


y_pred_rbf = svc_rbf.predict(X_test)


conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
print("Classification Report (RBF Kernel):")
print(classification_report(y_test, y_pred_rbf))


plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rbf, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - RBF Kernel")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


file_path = "weight-height.csv"
data = pd.read_csv("weight-height.csv")

X = data['Height']
y = data['Weight']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)



scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)



knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred_unscaled = knn.predict(X_test)

r2_unscaled = r2_score(y_test, y_pred_unscaled)
print(f"R2 Score (Unscaled Data): {r2_unscaled:.4f}")


knn.fit(X_train_minmax, y_train)

y_pred_minmax = knn.predict(X_test_minmax)

r2_minmax = r2_score(y_test, y_pred_minmax)
print(f"R2 Score (Normalized Data): {r2_minmax:.4f}")


knn.fit(X_train_standard, y_train)

y_pred_standard = knn.predict(X_test_standard)

r2_standard = r2_score(y_test, y_pred_standard)
print(f"R2 Score (Standardized Data): {r2_standard:.4f}")


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_unscaled = knn.predict(X_test)

r2_unscaled = r2_score(y_test, y_pred_unscaled)
print(f"R2 Score (Unscaled Data): {r2_unscaled:.4f}")


knn.fit(X_train_minmax, y_train)

y_pred_minmax = knn.predict(X_test_minmax)

r2_minmax = r2_score(y_test, y_pred_minmax)
print(f"R2 Score (Normalized Data): {r2_minmax:.4f}")


knn.fit(X_train_standard, y_train)

y_pred_standard = knn.predict(X_test_standard)

r2_standard = r2_score(y_test, y_pred_standard)
print(f"R2 Score (Standardized Data): {r2_standard:.4f}")


print("\nComparison of R2 Scores:")
print(f"R2 Score (Unscaled Data): {r2_unscaled:.4f}")
print(f"R2 Score (Normalized Data): {r2_minmax:.4f}")
print(f"R2 Score (Standardized Data): {r2_standard:.4f}")




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


file_path = "suv.csv"
data = pd.read_csv("suv.csv")

X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=20)
dt_classifier.fit(X_train_scaled, y_train)

y_pred = dt_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy Score: {accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = "suv.csv"
data = pd.read_csv("suv.csv")

X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_classifier_entropy = DecisionTreeClassifier(criterion='entropy', random_state=20)
dt_classifier_entropy.fit(X_train_scaled, y_train)

y_pred_entropy = dt_classifier_entropy.predict(X_test_scaled)

accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)
class_report_entropy = classification_report(y_test, y_pred_entropy)

print(f"Decision Tree with Entropy Criterion:")
print(f"Accuracy Score: {accuracy_entropy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_entropy}")
print(f"Classification Report:\n{class_report_entropy}")

dt_classifier_gini = DecisionTreeClassifier(criterion='gini', random_state=20)
dt_classifier_gini.fit(X_train_scaled, y_train)

y_pred_gini = dt_classifier_gini.predict(X_test_scaled)

accuracy_gini = accuracy_score(y_test, y_pred_gini)
conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)
class_report_gini = classification_report(y_test, y_pred_gini)

print(f"\nDecision Tree with Gini Criterion:")
print(f"Accuracy Score: {accuracy_gini:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_gini}")
print(f"Classification Report:\n{class_report_gini}")




