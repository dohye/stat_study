
## SGD regressor 
- beta 직접추정 / SGD class 생성 / sklearn 모델을 사용하여 모수 추정 및 예측을 실행해 본 후, 결과 비교

<br/>
<br/>

### 데이터 불러오기
중앙 플로리다 지역에서 연구를 위하여 포획된 15마리의 악어에서 2종류의 데이터 각각의 **체중**과 **주둥이 길이**를 구한 것  
  
단위는 체중(`pound`,y) , 주둥이의 길이(`inch`,X)


<br/>

```python
import numpy as np
  
X = np.array([3.87, 3.61, 4.33, 3.43, 3.81, 3.83, 3.46, 3.76, 3.50, 3.58, 4.19, 3.78, 3.71, 3.73, 3.78]).reshape(15,1)
y = np.array([4.87, 3.93, 6.46, 3.33, 4.38, 4.70, 3.50, 4.50,3.58, 3.64, 5.90, 4.43, 4.38, 4.42, 4.25]).reshape(15,1)
```

<br/>

* scatter plot
```python
import matplotlib.pyplot as plt
p = plt.scatter(X,y)
plt.show()
```

![sgd2](https://user-images.githubusercontent.com/37234822/60962714-d865c880-a349-11e9-9640-d7c7b2aebe3c.png)


<br/>
<br/>
<br/>

### 1. beta 직접 추정
  
```python
# X를 계획행렬로 변환  
one = np.ones((15,1))
des_X = np.hstack([one,X])
# des_X = np.array([[1.,3.87], [1.,3.61], [1., 4.33], [1.,3.43], [1.,3.81], [1.,3.83], [1.,3.46], [1.,3.76], [1.,3.50], [1.,3.58], [1.,4.19], [1.,3.78], [1.,3.71], [1.,3.73], [1.,3.78]])  
  
b = np.linalg.solve(np.matmul(des_X.T,des_X), np.matmul(des_X.T, y))
  
y_hat = np.matmul(des_X, b)
loss_ = 0.5 *np.sum((y_hat - y)**2)
```

```python
print("pred :", np.round(y_hat,4))
print("coef :", "[b : ", b[0],", W : ", b[1], "]")
print("loss : ", np.round(loss_,4))
```

```
pred : [4.8023], [3.9102], [6.3806], [3.2926], [4.5964], [4.665], [3.3955], [4.4249], [3.5328], [3.8073], [5.9002], [4.4935], [4.2533], [4.3219], [4.4935]  
coef : [b :  [-8.4760671] , W :  [3.43109822] ]  
loss :  0.0982
```
  
<br/>
<br/>
<br/>
  
### 2. SGD class
* 수치미분 함수 정의
```{python}
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val 
        
    return grad
```
<br/>

* 클래스 정의
```python
        
class linear_with_sgd:
    
    def __init__(self, input_size, output_size):
        
        self.params = dict()
        self.params['W'] = np.zeros((input_size, output_size)) #random.randn(input_size, output_size)
        self.params['b'] = np.zeros(output_size) 
  
    def set_data(self, x, t):  
  
        self.X = x
        self.y = t        
        
    def predict(self, x):
        
        W, b = self.params['W'], self.params['b']
        y = np.matmul(x,W) + b
        return y
        
    def loss(self, x, t):
        
        y = self.predict(x)
        loss = 0.5 * np.sum((y - t)**2)
        
        return loss
    
    def gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)
        
        grads=dict()
        grads['W'] = numerical_gradient(loss_W, self.params['W'])
        grads['b'] = numerical_gradient(loss_W, self.params['b'])
        
        return grads
  
    def get_coef(self):
        
        return self.params
    
    def training(self, learning_rate, ABSTOL):
        
        old_cost = 1E+32
        save_cost = list()
        
        for epoch in range(120000):
    
            grad = self.gradient(self.X, self.y)
            
            for key in ('W', 'b'):
                self.params[key] -= learning_rate * grad[key]
            cost = self.loss(self.X, self.y)
            save_cost.append(cost)
             
            if np.abs(old_cost - cost) < ABSTOL * old_cost:
                print("epoch :", epoch)
                break
            
            old_cost = cost
        return save_cost
```


##### 실행

  * learning_rate=`0.001`, epoch=`120,000`

```python
SGD = linear_with_sgd(1,1)
SGD.set_data(X, y)
SGD.training(0.001, 1E-8)
```

```python
print("pred :", np.round(SGD.predict(X), 4))
print("coef :", SGD.get_coef())
print("loss :", SGD.loss(X, y))
```

```
pred : [4.8005], [3.9127], [6.3711], [3.2981], [4.5956], [4.6639], [3.4005], [4.4249], [3.5371], [3.8103], [5.8931], [4.4932], [4.2542], [4.3225], [4.4932]
coef : {'b': array([-8.41352161]), 'W': array([[3.41446973]])}
loss : 0.09836405011617769
```

<br/>
<br/>
<br/>

### 3. sklearn_SGDRegressor 사용
  * learning_rate=`0.001`, epoch=`300,000`

```python
from sklearn import linear_model
  
clf = linear_model.SGDRegressor(alpha=0.001,n_iter=300000)
clf.fit(X,y)
clf.get_params(deep=True)
  
pred = clf.predict(X).reshape(15,1)
sk_loss = 0.5 *np.sum((pred - y)**2)
```


```python
print("pred :", np.round(pred,4))
print("W :", clf.coef_, ", b : ", clf.intercept_)
print("loss :", sk_loss)
```

```
pred : [4.7945], [3.9217], [6.3386], [3.3175], [4.5931], [4.6602], [3.4182], [4.4253], [3.5525], [3.821], [5.8687], [4.4924], [4.2574], [4.3245], [4.4924]
W : [3.35683203] , b :  [-8.19643677]
loss : 0.10060543404345937

```

<br/>
<br/>
<br/>


### 4. 결과 비교


y = [4.87, 3.93, 6.46, 3.33, 4.38, 4.70, 3.50, 4.50,3.58, 3.64, 5.90, 4.43, 4.38, 4.42, 4.25]



|**-** |**beta** |**SGD class** |**sklearn** |
|:--------:|:--------:|:--------:|:--------:|
|**epoch** |**-** |**120,000**  |**300,000** |
|**lr** |**-**  |**0.001** |**0.001**  |
|**W** |**3.4311**  |**3.4266** |**3.3565**  |
|**b** |**-8.4761** |**-8.4591** |**-8.1958** |
|**pred** |**[4.80, 3.91, 6.38, 3.29, 4.60, 4.67,  3.40, 4.42, 3.53, 3.81,  5.90, 4.49, 4.25, 4.32, 4.49]** |**[4.80, 3.91 6.38, 3.29, 4.60, 4.66, 3.40, 4.42, 3.53,  3.81, 5.90, 4.49, 4.25, 4.32, 4.49]** |**[4.80,  3.92, 6.34,  3.32, 4.59, 4.66, 3.42, 4.42, 3.55, 3.82, 5.87, 4.49, 4.26,  4.32, 4.49]**  |
|**loss** |**0.0982** |**0.0983** |**0.1006** |
