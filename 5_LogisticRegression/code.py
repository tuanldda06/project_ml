import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Titanic.csv')
X = df['PassengerID'].values.astype(float)
y = df['Survived'].values.reshape(-1,1).astype(float)

N = X.shape[0]
np.random.seed(34)
split = np.random.permutation(N)
idx = int(0.8*N)
train , test = split[:idx] , split[idx:]
X_train_raw , y_train = X[train], y[train]
X_test_raw , y_test = X[test] , y[test]

X_mean = X_train_raw.mean(axis = 0 , keepdims = True)
X_std = X_train_raw.std(axis = 0 , keepdims = True)
