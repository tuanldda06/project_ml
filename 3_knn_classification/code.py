import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
def euclid_distance(x1,x2) :
    return np.sqrt(np.sum((x1-x2)**2))

def knn_classification(x_train , y_train , x_test , k = 3) :
    distance =[]
    for i in range(len(x_train)) :
        dist = euclid_distance(x_train[i],x_test)
        distance.append((dist , y_train[i]))
    distance.sort(key=lambda x : x[0])
    neighbors = distance[:k]

    output = [neighbor[1] for neighbor in neighbors]
    predict = Counter(output).most_common(1)[0][0] # trả về cái label nào mà được xuất hiện nhiều nhất 
    return predict

def knn_accuracy(x_train , y_train , x_test , y_test , k = 3) :
    correct = 0
    for i in range(len(x_test)):                     # duyệt theo số mẫu Test
        predict = knn_classification(x_train , y_train , x_test[i] , k)  # từng điểm
        if predict == y_test[i] :
            correct+=1
    return correct / len(x_test)                     # chia cho số mẫu Test


df = pd.read_csv('Breast Cancer Wisconsin (Diagnostic).csv')
df = df.drop(columns=['id','Unnamed: 32'])
X = df.drop(columns=['diagnosis']).values.astype(float)
y = df['diagnosis'].map({'M' : 0 , 'B' :1}).values
np.random.seed(10)
N = X.shape[0]
index = np.random.permutation(N)
split = int(0.8*N)
train , test = index[:split] , index[split:]
X_train , y_train = X[train] , y[train]
X_test , y_test = X[test] , y[test]





train_acc_list = [] # để ứng với mỗi k , để xem nó hiểu dữ liệu đã được train như thế nào (overfit , underfit , ....)
valid_acc_list = []# dùng chính là để chọn k tối ưu.
#Thông thường thì train_acc_list >= test_acc_list do nó đã được học qua dữ liệu train từ trước , còn test thì chưa
    


k_list = [1,3,5,8,9,11,13,15,19,21,23,26,28,30]
best_k = None
best_acc = -1.0

for k in k_list :
    acc_train = knn_accuracy(X_train , y_train , X_train , y_train , k )# để ứng với mỗi k , để xem nó hiểu dữ liệu đã được train như thế nào (overfit , underfit , ....)
    acc_test = knn_accuracy(X_train , y_train , X_test , y_test , k )# dùng chính là để chọn k tối ưu.
    train_acc_list.append(acc_train)
    valid_acc_list.append(acc_test)
    
    if acc_test > best_acc:   
        best_acc = acc_test
        best_k = k

print(f"\nBest k = {best_k}, train accuracy = {best_acc:.4f}")

plt.plot(k_list, train_acc_list, marker='o', label='Train acc')
plt.plot(k_list, valid_acc_list, marker='s', label='Valid acc')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
