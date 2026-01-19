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
one = np.ones(X_train.shape[0],1)
Xbar = np.concatenate((one , X_train),axis = 1)

w = np.random.randn(Xbar.shape[1],1) * 0.01
eta = 1e-6
epochs = 2000000
m = X_train.shape[0]
lamda = 8.0
for _ in range(epochs) :
    y_pred = Xbar @ w
    loss = y_pred - y_train

