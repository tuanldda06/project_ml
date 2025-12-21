import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def euclid_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_regression(x_train, y_train, x_test, k=3):
    distance = []
    for i in range(len(x_train)):
        dist = euclid_distance(x_train[i], x_test)
        distance.append((dist, y_train[i]))
    distance.sort(key=lambda x: x[0])
    neighbors = distance[:k]
    output = [neighbor[1] for neighbor in neighbors]
    predict = sum(output) / len(output)
    return predict

def knn_mse(x_train, y_train, x_test, y_test, k=3):
    mse = 0.0
    for i in range(len(x_test)):
        pred = knn_regression(x_train, y_train, x_test[i], k)
        mse += (pred - y_test[i]) ** 2
    mse /= len(x_test)
    return mse

df = pd.read_csv('predict_pricehouse.csv')
features = [
    'sqft_living',
    'bedrooms',
    'bathrooms',
    'floors',
    'view',
    'condition',
    'sqft_basement',
    'yr_built',
    'waterfront'
]
X = df[features].values.astype(float)
y = df['price'].values.astype(float)

np.random.seed(10)
N = X.shape[0]
idx = np.random.permutation(N)
split = int(0.8 * N)
train, test = idx[:split], idx[split:]
X_train_raw, y_train = X[train], y[train]
X_test_raw,  y_test  = X[test],  y[test]

# Chuẩn hóa theo từng cột
X_mean = X_train_raw.mean(axis=0, keepdims=True)
X_std  = X_train_raw.std(axis=0, keepdims=True)

X_train = (X_train_raw - X_mean) / X_std
X_test  = (X_test_raw  - X_mean) / X_std

k_list = [1,3,4,7,9,11,13,14,15,18,19,20,21,23,25,27]
best_k = None
best_mse = float('inf')
train_mse_list = []
valid_mse_list = []

for k in k_list:
    mse_train = knn_mse(X_train, y_train, X_train, y_train, k)  # dùng train
    mse_valid = knn_mse(X_train, y_train, X_test,  y_test,  k)  # dùng test

    train_mse_list.append(mse_train)
    valid_mse_list.append(mse_valid)

    #print(f"k = {k:2d} | train MSE = {mse_train:.2f} | test MSE = {mse_valid:.2f}")

    if mse_valid < best_mse:   # MSE càng nhỏ càng tốt
        best_mse = mse_valid
        best_k = k

#print(f"\nBest k = {best_k}, với test MSE = {best_mse:.2f}")

plt.plot(k_list, train_mse_list, marker='o', label='Train MSE')
plt.plot(k_list, valid_mse_list, marker='s', label='Test MSE')
plt.xlabel('k')
plt.ylabel('MSE')
plt.legend()
plt.show()
