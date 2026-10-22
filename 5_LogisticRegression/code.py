import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic_train.csv')
df['Sex'] = df['Sex'].map({'male':0 , 'female':1})
df['Age'] = df['Age'].fillna(df['Age'].median())
features = ['Pclass','Sex' ,'Age']
X = df[features].values.astype(float)
y = df['Survived'].values.reshape(-1,1).astype(float)

N = X.shape[0]
split = np.random.permutation(N)
idx = int(0.8*N)
train , test = split[:idx] , split[idx:]
X_train_raw , y_train = X[train] , y[train]
X_test_raw ,y_test = X[test] , y[test]
X_mean = X_train_raw.mean(axis = 0 , keepdims =True)#keepdims=True: giữ nguyên số chiều → kết quả có shape (1, n_features) thay vì (n_features,)
X_std = X_train_raw.std(axis = 0 , keepdims = True)#keepdims=True: giữ nguyên số chiều → kết quả có shape (1, n_features) thay vì (n_features,)
X_train = (X_train_raw-X_mean) / X_std
X_test = (X_test_raw - X_mean) /X_std


one = np.ones((X_train.shape[0] , 1))
Xbar = np.concatenate((one , X_train) , axis = 1)

w = np.zeros((Xbar.shape[1],1)) # phải dùng Xbar do Xbar đã thêm cái w0 , còn cái X_train thì nó chưa có w0 nên không dùng để train được , thuộc tính thì là cột
lrn_rate = 0.001
epochs = 100
losses = []
def sigmoid(z):
    return 1 /( 1 + np.exp(-z))#tính giá trị lũy thừa của số e
for i in range(epochs) :
    y_pred = sigmoid(Xbar @ w) # Hàm dự đoán ( sẽ đưa ra giá trị ở trong khoảng [0,1])
    loss = -np.sum(y_train * np.log(y_pred + 1e-9) + (1-y_train)*np.log(1-y_pred + 1e-9))
    losses.append(loss)
 
 # Tối ưu w để mô hình tốt hơn
    gradient = Xbar.T @ (y_pred - y_train)
    w = w - lrn_rate * gradient

    #Với dữ liệu train này thì :Xbar có kích thước (Nx4) , Xbar.T có kích thước (4xN) , w có kích thước(4X1)


y_train_prob = sigmoid(Xbar @ w)#Tính xác suất dự đoán khi mà đã train được cái w cho cái mô hình 
y_train_final = (y_train_prob >=0.5).astype(float)#Là phép so sánh. Nếu xác suất >=0.5, nó trả về True, ngược lại trả về False. Con số 0.5 chính là cái ngưỡng (threshold) để phân loại.
acc_train = np.mean(y_train_final==y_train) #Tính độ chính xác (Accuracy).


one_test = np.ones((X_test.shape[0] ,1))
Xbar_test = np.concatenate((one_test, X_test),axis = 1)

y_test_prob = sigmoid(Xbar_test @ w)#prob là viết tắt của  Probability, nghĩa là Xác suất.
y_test_final = (y_test_prob >=0.5).astype(float)
acc_test = np.mean(y_test_final == y_test)

print("Accuracy train =", acc_train)
print("Accuracy test  =", acc_test)


plt.plot(range(epochs),losses,color = 'Blue' , label = 'Train loss')
plt.xlabel('So vong lap')
plt.ylabel('Tong loi')
plt.title('Quá trình Gradient Descent giảm lỗi')
plt.legend()
plt.show() 
