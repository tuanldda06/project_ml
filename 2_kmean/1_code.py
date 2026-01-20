from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd

np.random.seed(10)

def kmeans_init(X,K) :
    return X[np.random.choice(X.shape[0],K ,replace = False)]

def kmeans_assign(X,center) :
    K = cdist(X,center)
    return np.argmin(K,axis =1)#dùng axis=1 ở np.argmin vì ta cần lấy min theo từng hàng (mỗi hàng là một điểm), để tìm tâm gần nhất cho MỖI điểm .Còn axis = 0 thì nó so sánh theo cột của từng cái một với nhau , nên không đúng

def kmeans_update(X,label , K) :
    center = np.zeros((K,X.shape[1]))
    for k in range(K) :
        Xk = X[label == k ,:]
        if Xk.shape[0] > 0 :
            center[k, :] = np.mean(Xk,axis = 0)
        else :
            center[k, :] = X[np.random.randint(0 , X.shape[0])]
    return center

def has_conveged(center,new_center,tol = 1e-6) :
    return np.allclose(center , new_center , rtol = 0 , atol = tol)

def kmeans_main(X,K) :
    center = kmeans_init(X,K)
    while True :
        label = kmeans_assign(X,center)
        new_center = kmeans_update(X,label , K)
        if has_conveged(center,new_center,tol = 1e-6) :
            break
        center = new_center
    return center , label

df = pd.read_csv('Khách Hàng Mall.csv')
X = df[['Age','Annual Income (k$)','Spending Score (1-100)']].values.astype(float)
K =3
center , label = kmeans_main(X,K)
SSE = 0
for i in range(len(X)) :
    diff = X[i] - center[label[i]]
    SSE += np.sum(diff**2)
print(f'SSE = {SSE}')
plt.scatter(X[:,0] , X[:,1] , c = label , s =1)
plt.scatter(center[:,0],center[:,1] , c = 'red' , marker = 'X' , s = 200)
#[:,0] nghĩa là lấy tất cả các cột của hàng 0 , [:,1] nghĩa là lấy tất cả các cột của hàng 1
plt.show()
