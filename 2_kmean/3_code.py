from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd


def kmeans_init(X,K) :
    if K > X.shape[0] :
        print('K phai be hon so hang cua X')
    else :
        center = np.zeros((K,X.shape[1]))
        idx = np.random.choice(X.shape[0],K , replace = False)
        center = X[idx]
    return center


def kmean_asssign(X,center) :
    D = cdist(X,center)
    return np.argmin(D , axis = 1)



def kmean_update_online(center , count , x , k ) :#Mỗi điểm x sẽ được gán vào cụm gần nhất k.
    count[k] = count[k] + 1#Vì cụm k vừa nhận thêm 1 điểm mới, nên tăng bộ đếm: counts[k] += 1.
    eta = 1.0/count[k]
    center[k,:] = center[k,:] + eta * ( x - center[k,:])##center[k,:] là tâm của cụm k (lấy toàn bộ tọa độ), vì mỗi hàng trong center tương ứng một cụm.
    return center , count


def has_converge(center , new_center , tol = 1e-6) :
    return np.allclose(center , new_center , rtol = 0.0 ,atol = tol)



def kmeans_main(X,K) :
    center = kmeans_init(X,K)
    count = np.zeros(K , dtype = np.int64)#Tạo mảng đếm độ dài K và counts[k] = số điểm đã được gán vào cụm k (dùng để tính learning rate 1/counts[k]).
    while True :
        old_center = center.copy()
        idx = np.random.permutation(X.shape[0])# xào trộn để online ổn định hơn
        for i in idx :
            D = cdist(X[i].reshape(1,-1),center)#tính khoảng cách từ điểm i tới mọi tâm.
            #reshape là để biến “một điểm” thành mảng 2D (1, d) đúng format mà cdist yêu cầu.
            k = np.argmin(D,axis = 1)[0]#Lấy chỉ số cụm có khoảng cách nhỏ nhất → cụm gần nhất với X[i].

            center , count = kmean_update_online(center , count , X[i] , k )
        if has_converge(old_center , center , tol = 1e-6) :
            break
    label = kmean_asssign(X,center)#gán nhãn cho toàn bộ dữ liệu theo tâm cuối
    return center , label#trả về tâm và nhãn.
df = pd.read_csv('Khách Hàng Mall.csv')
X = df[['Age','Annual Income (k$)','Spending Score (1-100)']].values.astype(float)
K =3
center , label = kmeans_main(X,K)
SSE =0
for i in range(len(X)) :
    diff = X[i] - center[label[i]]
    SSE += np.sum(diff**2)
print('SSE=' , SSE)

plt.scatter(X[:,0] , X[:,1] , c = label , s =1)
plt.scatter(center[:,0] , center[:,1] , c = 'red' , marker = 'X' , s= 200)
plt.show()

