n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    print(f"Step 1: Processing n = {n}")

    print(f"Step 2: Result of calculation for n = {n} is {200000}")

    print(f"Step 3: Action completed for n = {n}")

import numpy as np

n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)

    dice_sum = dice1 + dice2

    print(f"Sum of dice for n = {n}: {dice_sum}")




    n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]


    for n in n_values:

        dice1 = np.random.randint(1, 7, n)
        dice2 = np.random.randint(1, 7, n)


        dice_sum = dice1 + dice2


        h, h2 = np.histogram(dice_sum, bins=range(2, 14))


        print(f"Frequencies for n = {n}: {h}")
        print(f"Bin edges for n = {n}: {h2}")


        import numpy as np
        import matplotlib.pyplot as plt


        n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]


        for n in n_values:

            dice1 = np.random.randint(1, 7, n)
            dice2 = np.random.randint(1, 7, n)


            dice_sum = dice1 + dice2


            h, h2 = np.histogram(dice_sum, bins=range(2, 14))


            plt.bar(h2[:-1], h / n)
            plt.title(f"Histogram of Dice Sums for n = {n}")
            plt.xlabel("Sum of Dice")
            plt.ylabel("Frequency")
            plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


file_path = 'weight-height.csv'
data = pd.read_csv(file_path)


print(data.head())


plt.figure(figsize=(8, 6))
plt.scatter(data['Height'], data['Weight'], alpha=0.7, color='blue')
plt.title('Scatter Plot of Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.grid()
plt.show()


X = data['Height'].values.reshape(-1, 1)
y = data['Weight'].values


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


plt.figure(figsize=(8, 6))
plt.scatter(data['Height'], data['Weight'], alpha=0.7, color='blue', label='Data Points')
plt.plot(data['Height'], y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.grid()
plt.show()


print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")


mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


