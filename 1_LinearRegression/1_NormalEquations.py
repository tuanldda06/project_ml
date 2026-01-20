import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
y = df['price'].values.reshape(-1,1).astype(float)
N = X.shape[0]
idx = np.random.permutation(N)
split = int(0.8*N)
train , test = idx[:split] , idx[split:]
X_train_raw , y_train = X[train] , y[train]
X_test_raw , y_test = X[test] , y[test]

X_mean = X_train_raw.mean(axis = 0 , keepdims =True)#keepdims=True: giữ nguyên số chiều → kết quả có shape (1, n_features) thay vì (n_features,)
X_std = X_train_raw.std(axis = 0 , keepdims = True)#keepdims=True: giữ nguyên số chiều → kết quả có shape (1, n_features) thay vì (n_features,)
X_train = (X_train_raw-X_mean) / X_std
X_test = (X_test_raw - X_mean) /X_std

one = np.ones((X_train.shape[0],1))# 2 dấu () đảm bảo shape (m,1) để np.concatenate để nối lại với nhau
Xbar = np.concatenate((one , X_train),axis = 1)

A = Xbar.T @ Xbar
b = Xbar.T @ y_train
w = np.linalg.pinv(A) @ b

y_train_final = Xbar @ w
mse = np.mean((y_train_final - y_train)**2)
mae = np.mean(np.abs(y_train_final - y_train))


one_test = np.ones((X_test.shape[0],1))
Xbar_test = np.concatenate((one_test , X_test),axis = 1)
y_test_final = Xbar_test @ w

mse_test = np.mean((y_test_final - y_test)**2)
mae_test = np.mean(np.abs(y_test_final - y_test))


print("MSE train =", mse)
print("MAE train =", mae)
print("MSE test  =", mse_test)
print("MAE test  =", mae_test)

n = 500
y_true_plot = y_test[:n]
y_pred_plot = y_test_final[:n]

plt.scatter(y_true_plot,y_pred_plot,color = 'green',label = 'Test')
plt.scatter(y_train[:n],y_train_final[:n],color = 'red',label = 'Train')

mn = min(y_true_plot.min(),y_pred_plot.min())
mx = max(y_true_plot.max(),y_pred_plot.max())

x_line = np.array([[mn],[mx]])
y_line = x_line
plt.plot(x_line , y_line , label = 'y = x')
plt.xlabel('Gia that (y_true)')
plt.ylabel('Gia du doan (y_pred)')
plt.legend()
plt.show()

