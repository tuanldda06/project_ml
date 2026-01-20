from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd


np.random.seed(10)


def kmeans_init(X,K) :
    if K > X.shape[0] :
        print('K phai be hon so hang cua X')
    else :
        center = np.zeros((K,X.shape[1]))# ban đầu nó bằng 0 hết
        center[0]= X[np.random.choice(X.shape[0])] # chọn ngẫu nhiên center ban đầu 
        for k in range(1,K) :# lặp để tìm center cho các K-1 tâm còn lại
            distances = np.min(cdist(X,center[:k]),axis =1)# trong mỗi lần lặp, với từng điểm dữ liệu, chỉ giữ lại khoảng cách nhỏ nhất tới các tâm đã chọn (tâm gần nhất)
            probabilities = distances**2 / (np.sum(distances **2) + 1e-6)# numpy sẽ vector hóa để áp dụng công thức như kia mà không cần sử dụng tới vòng lặp
            center[k] = X[np.random.choice(X.shape[0],p = probabilities)]
    return center

def kmean_assign(X,center) :
    D = cdist(X,center) 
    return np.argmin(D,axis=1)#dùng axis=1 ở np.argmin vì ta cần lấy min theo từng hàng (mỗi hàng là một điểm), để tìm tâm gần nhất cho MỖI điểm .Còn axis = 0 thì nó so sánh theo cột của từng cái một với nhau , nên không đúng

def kmeans_update(X,label ,K) :
    center = np.zeros((K,X.shape[1]))
    for k in range(K) :
        Xk =X[label ==k,:]#Lấy tất cả điểm nằm trong cụm k → mỗi điểm gồm đầy đủ tọa độ x và y , còn [label ==k,:] là lấy tất cả các hàng mà label == k
        if Xk.shape[0] > 0 :
            center[k,:] = np.min(Xk,axis = 0)
        else :
            center[k,:] = X[np.random.randint(0,X.shape[0])]#trả về một số nguyên trong [0, n)
    return center 

def has_converge(center , new_center , tol = 1e-6) :
    return np.allclose(center,new_center,rtol = 0.0 , atol = tol)
def kmeans_main(X,K):
    center = kmeans_init(X,K)
    while True :
        label = kmean_assign(X,center)
        new_center = kmeans_update(X,label ,K)
        if has_converge(center , new_center , tol = 1e-6):
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
