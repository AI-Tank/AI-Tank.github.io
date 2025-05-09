```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
```


```python
# 데이터셋 불러오기
iris = load_iris()
X = iris.data  # 특징 데이터 (4차원)
y = iris.target  # 레이블 (0, 1, 2)

# 학습용과 테스트용으로 데이터 분리 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"학습 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")
```

    학습 데이터 크기: (120, 4)
    테스트 데이터 크기: (30, 4)
    


```python
def forward(input, weights, bias) :
    output = np.dot(input, weights) + bias
    return output

def relu(x) : 
    x = (x + np.absolute(x)) / 2
    return x

def softmax(x) :
    y_pred = np.zeros(x.shape)
    for i in range(x.shape[0]) : 
        x[i] = x[i] - np.max(x[i])
        y_pred[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
    return y_pred

def cross_entropy_loss(y_pred, y_true) :
    return -np.sum(y_true * np.log(y_pred), axis = 0)

def onehot(y, value = 3) : 
    y_pred = np.zeros((y.shape[0], value))
    for i in range(y_pred.shape[0]) : 
        if y[i] == 0 :
            y_pred[i, 0] = 1
        if y[i] == 1 : 
            y_pred[i, 1] = 1
        if y[i] == 2 :
            y_pred[i, 2] = 1
    return y_pred
```


```python
np.random.seed(42)
w1 = np.random.rand(4, 10)
b1 = np.zeros((1, 10))
w2 = np.random.rand(10, 3)
b2 = np.zeros((1, 3))
```


```python
epochs = 500
for i in range(epochs) : 
    y = np.sum(y_train, axis = 0) / y_train.shape[0]

    z1 = forward(X_train, w1, b1)
    a1 = relu(z1)
    z2 = forward(a1, w2, b2)
    a2 = softmax(z2)
    y_train_onehot = onehot(y_train)
    loss = cross_entropy_loss(a2, y_train_onehot) / y_train.shape[0]

    dz2 = a2 - y_train_onehot
    dw2 = np.dot(a1.transpose(), dz2) / y_train.shape[0]
    db2 = np.sum(dz2, axis = 0) / y_train.shape[0]

    da1 = np.dot(dz2, w2.transpose())
    dz1 = da1 * ((np.absolute(z1) + z1) / (2 * z1))
    dw1 = np.dot(X_train.transpose(), dz1) / y_train.shape[0]
    db1 = np.sum(dz1, axis = 0) / y_train.shape[0]

    w1 = w1 - 0.1 * dw1
    w2 = w2 - 0.1 * dw2
    b1 = b1 - 0.1 * db1
    b2 = b2 - 0.1 * db2
```


```python
from sklearn.metrics import f1_score

# 데이터 스플릿으로 y_valid와 모델 예측으로 y_pred를 구한 후 실행
# 모델 검정이 없다면 y_true값으로 y_valid 대체

y_pred = softmax(forward(relu(forward(X_test, w1, b1)), w2, b2))
for i in range(y_pred.shape[0]) : 
    if y_pred[i][0] > y_pred[i][1] :
        if y_pred[i][0] > y_pred[i][2] : 
            y_pred[i] = 0
        else : y_pred[i] = 2
    elif y_pred[i][1] > y_pred[i][2] :
        y_pred[i] = 1
    else : y_pred[i] = 2
y_pred = np.sum(y_pred, axis = 1) / 3

f1 = f1_score(y_pred, y_test, average='weighted')
print(f1)
```

    1.0
    


```python
print(y_pred)
print(y_test)
```

    [1. 0. 2. 1. 1. 0. 1. 2. 1. 1. 2. 0. 0. 0. 0. 1. 2. 1. 1. 2. 0. 2. 0. 2.
     2. 2. 2. 2. 0. 0.]
    [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]