## Newton-Raphson Method



<img src="https://latex.codecogs.com/svg.latex?f(x)=0"> 의 해를 구하는 방법 중 하나로, 반복 계산을 통해 더 가까운 근사값으로 수렴하는 방법  
또한, 일반적으로 함수는 극점에서 최소값, 최대값을 가지므로 <img src="https://latex.codecogs.com/svg.latex?f'(x)=0">인 <img src="https://latex.codecogs.com/svg.latex?x">를 newton raphson method로 구한 후 <img src="https://latex.codecogs.com/svg.latex?f(x)">에 대입하면 최대, 최소값을 구할 수 있음

<br/>

![nr](https://user-images.githubusercontent.com/37234822/61104596-e6365d80-a4b1-11e9-8115-c4a165f18ff1.png)

<br/>

![nr0](https://user-images.githubusercontent.com/37234822/61104588-e3d40380-a4b1-11e9-8bf9-e44138d75b76.JPG)

<br/>

![nr1](https://user-images.githubusercontent.com/37234822/61104592-e5053080-a4b1-11e9-962f-2d548db1a763.JPG)

<br/>

### 1. <img src="https://latex.codecogs.com/svg.latex?f(x)=0"> 의 해를 구하기


* <img src="https://latex.codecogs.com/svg.latex?f(x)"> 함수 지정
```python
def f(x):
    y = x**2 - 4
    return y
```

<br/>

* <img src="https://latex.codecogs.com/svg.latex?f'(x)"> 함수 정의
```python
def derivative(f, x):
    h = 0.000001   
    derivative = (f(x + h) - f(x)) / h
    return derivative
```

<br/>

* newton-raphson 함수

```python
import numpy as np
def newton_raphson(x0, ABSTOL, iterate_num):
    
    result = list()
    result.append(x0)
    
    for i in range(iterate_num):
        
        if derivative(f, x0) == 0:
            print("can't be differentiated")
        
        x = x0 - f(x0)/derivative(f, x0)
        result.append(x)       
        if np.abs(x0-x) < ABSTOL:
            print("result : ", result)
            break
        
        x0 = x
        i += 1
        
    if np.abs(f(x0)) > ABSTOL: 
        print("do not converge within the number of iterations")
```

<br/>

* 실행 ( 초기값, 오차허용도, 반복수 지정 )
 
```python
#지정반복수 내에 수렴하지 않는 경우
newton_raphson(10, 5E-08, 5)
```

```
do not converge within the number of iterations
```

<br/>

```python
#지정반복수 내에 수렴하는 경우
newton_raphson(10, 5E-08, 20)
```

```
result :  [10, 5.2000002368553275, 2.98461569884197, 2.162411009581745, 2.006099093168112, 2.0000092729796166, 2.000000000023817, 2.0]
```

<br/>

```python
#초기값을 음수로 지정하면 그에 가까운 해를 찾게됨
newton_raphson(-10, 5E-08, 20)
```


```
result :  [-10, -5.199999755960004, -2.98461506767821, -2.162410560111547, -2.0060989883892253, -2.00000926962557, -2.0000000000191642, -2.0]
```

<br/>
<br/>


### 2. <img src="https://latex.codecogs.com/svg.latex?f(x)"> 의 최소, 최대값 구하기

* <img src="https://latex.codecogs.com/svg.latex?f(x)">  함수지정
```{python}
def f(x):
    y = x**2 - 4
    return y
```
<br/>

* <img src="https://latex.codecogs.com/svg.latex?g(x)">  함수지정 (<img src="https://latex.codecogs.com/svg.latex?g(x)=f'(x)"> )

```python
def g(x):
    y = 2*x
    return y
```
<br/>


* <img src="https://latex.codecogs.com/svg.latex?f'(x)">

```python
def derivative(f, x):
    h = 0.000001   
    derivative = (f(x + h) - f(x)) / h
    return derivative
```
<br/>

* newton-raphson 함수

```python
import numpy as np
def newton_raphson_2(x0, ABSTOL, iterate_num):
    
    result = list()
    result.append(x0)
    
    for i in range(iterate_num):
        
        if derivative(g, x0) == 0:
            print("can't be differentiated")
        
        x = x0 - g(x0)/derivative(g, x0)
        result.append(x)       
        if np.abs(x0-x) < ABSTOL:
            print(" x_list :", result, "\n","max/min value : ", f(result[-1]))
            break
        
        x0 = x
        i += 1
        
    if np.abs(g(x0)) > ABSTOL: 
        print("do not converge within the number of iterations")
        
```


* 실행 ( 초기값, 오차허용도, 반복수 지정 )
 
```python
newton_raphson_2(5, 5E-08, 10)
```

```
x_list : [5, 6.988898348936345e-10, 0.0] 
max/min value :  -4.01
```
