import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10, 10, 100)


y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3


plt.figure(figsize=(8, 6))
plt.plot(x, y1, label="y = 2x + 1", color="blue")
plt.plot(x, y2, label="y = 2x + 2", color="green")
plt.plot(x, y3, label="y = 2x + 3", color="red")


plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Lines: y = 2x + 1, y = 2x + 2, y = 2x + 3")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()





x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


print(x)




y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])


print(y)




inches_to_cm = 2.54
pounds_to_kg = 0.453592


length_cm = length * inches_to_cm
weight_kg = weight * pounds_to_kg

print("Lengths in cm:", length_cm[:5])
print("Weights in kg:", weight_kg[:5])



mean_length_cm = np.mean(length_cm)
mean_weight_kg = np.mean(weight_kg)


print("Mean length (cm):", mean_length_cm)
print("Mean weight (kg):", mean_weight_kg)


import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))
plt.hist(length_cm, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Lengths')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()


import numpy as np


A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])


A_inv = np.linalg.inv(A)


identity_1 = np.dot(A, A_inv)
identity_2 = np.dot(A_inv, A)


print("Inverse of A:\n", A_inv)
print("A * A_inv:\n", identity_1)
print("A_inv * A:\n", identity_2)



