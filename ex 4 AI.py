import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes  # sklean data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes(as_frame=True)
print(data.keys())

print(data.DESCR)

df = data['frame']
print(df)

plt.hist(df["target"], 30)
plt.xlabel("target")
plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(df['age'], df['target'])  # age and s6 instead of bmi n s5
plt.xlabel('age')
plt.ylabel('target')

plt.subplot(1, 2, 2)
plt.scatter(df['s6'], df['target'])
plt.xlabel('s6')
plt.ylabel('target')
plt.show()

X = pd.DataFrame(df[['age', 's6']], columns=['age', 's6'])

y = df['target']
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)
print(X_train.shape)
print(X_test.shape)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("50_Startups.csv")


print("Dataset Columns:")
print(data.columns)


data = pd.get_dummies(data, columns=['State'], drop_first=True)


correlation_matrix = data.corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


explanatory_vars = ['R&D Spend', 'Administration', 'Marketing Spend']
for var in explanatory_vars:
    plt.figure(figsize=(6, 4))
    plt.scatter(data[var], data['Profit'], alpha=0.7)
    plt.title(f"{var} vs Profit")
    plt.xlabel(var)
    plt.ylabel("Profit")
    plt.show()


X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("Training RMSE:", train_rmse)
print("Training R2:", train_r2)
print("Testing RMSE:", test_rmse)
print("Testing R2:", test_r2)


plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_test_pred, alpha=0.7, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
plt.title("Actual vs Predicted Profit (Testing Data)")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

data = pd.read_csv("Auto.csv")

X = data.drop(columns=['mpg', 'name', 'origin'])
y = data['mpg']

X = X.select_dtypes(include=[np.number])
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

alphas = np.logspace(-4, 4, 100)
ridge_r2 = []
lasso_r2 = []

for alpha in alphas:

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_r2.append(r2_score(y_test, ridge_pred))


    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_r2.append(r2_score(y_test, lasso_pred))


optimal_ridge_alpha = alphas[np.argmax(ridge_r2)]
optimal_lasso_alpha = alphas[np.argmax(lasso_r2)]
print(f"Optimal Ridge Alpha: {optimal_ridge_alpha}, Best Ridge R2: {max(ridge_r2)}")
print(f"Optimal Lasso Alpha: {optimal_lasso_alpha}, Best Lasso R2: {max(lasso_r2)}")

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_r2, label='Ridge Regression R2', color='blue')
plt.plot(alphas, lasso_r2, label='Lasso Regression R2', color='green')
plt.axvline(optimal_ridge_alpha, linestyle='--', color='blue', label=f'Optimal Ridge Alpha: {optimal_ridge_alpha:.4f}')
plt.axvline(optimal_lasso_alpha, linestyle='--', color='green', label=f'Optimal Lasso Alpha: {optimal_lasso_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('R2 Scores for Ridge and Lasso Regression')
plt.legend()
plt.show()

print("Ridge Regression Best Alpha:", optimal_ridge_alpha)
print("Lasso Regression Best Alpha:", optimal_lasso_alpha)

