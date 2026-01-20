import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('predict_pricehouse.csv')
features = ['sqft_living',
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


X_mean = X_train_raw.mean(axis = 0 , keepdims = True)
X_std = X_train_raw.std(axis = 0 , keepdims = True)
X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean)/X_std
one = np.ones((X_train.shape[0],1))# 2 dấu () đảm bảo shape (m,1) để np.concatenate để nối lại với nhau
Xbar = np.concatenate((one , X_train),axis = 1)

w = np.random.randn(Xbar.shape[1],1) * 0.01
eta = 1e-4
epochs = 50000
m = X_train.shape[0]
lamda = 8.0
for _ in range(epochs) :
    y_pred = Xbar @ w
    loss = y_pred - y_train
    gradient_1 = (1/m) *(Xbar.T @ loss)
    gradient_2 = np.zeros_like(w)
    gradient_2[1:] = (lamda/m) * np.sign(w[1:])
    gradient = gradient_1+ gradient_2
    w = w - eta*gradient

y_train_final = Xbar @ w
mse = np.mean((y_train_final - y_train)**2)
mae = np.mean(np.abs(y_train_final-y_train))

one_test = np.ones((X_test[0],1))
Xbar_test = np.concatenate((one_test,X_test),axis = 1)
y_test_final = Xbar_test @ w
mse_test = np.mean((y_test_final - y_test)**2)
mae_test = np.mean(np.abs(y_test_final-y_test))

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
plt.plot(x_line , y_line , label = 'Fit for train')
plt.xlabel('Gia tri dung')
plt.ylabel('Gia tri du doan')
plt.legend()
plt.show()


