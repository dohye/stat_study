## 2차 테일러 정리
181021_stat_class
<br/>

![taylor1](https://user-images.githubusercontent.com/37234822/61225159-fc1a7b80-a75a-11e9-95ee-b8b73ffd7990.JPG)

<br/>

![taylor2](https://user-images.githubusercontent.com/37234822/61225161-fc1a7b80-a75a-11e9-94e7-52212d7798f7.JPG)

<br/>

![taylor3](https://user-images.githubusercontent.com/37234822/61225162-fc1a7b80-a75a-11e9-9353-72818f2d5749.JPG)

<br/>



### 1. <img src="https://latex.codecogs.com/svg.latex?f(x)=e^x">
```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2, 2, 100)
y = np.exp(x)
y1 = 1+x #1차 테일러
y2 = 1+x+(x*x/2) # 2차 테일러  
  
fig = plt.figure()
plt.plot(x,y,'k', label="$e^x$")
plt.title("$y = e^x$")
plt.plot(x, y1,'r--', label="taylor_1")
plt.plot(x, y2,'g--', label="taylor_2")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()
# fig1.savefig('fig1.png')
```
![taylor4](https://user-images.githubusercontent.com/37234822/61225163-fcb31200-a75a-11e9-8d46-1edd86e8841e.png)

<br/>
<br/>

----

### 2. <img src="https://latex.codecogs.com/svg.latex?f(x)=logx">
```python
x = np.linspace(-1,3,100)
logy = np.log(x)
logy1 = x-1 # 1차 테일러
logy2 = (x-1)-((x-1)*(x-1)/2) # 2차 테일러  
  
fig2 = plt.figure()
plt.title("$y = logx$")
plt.plot(x,logy,'k', label="$log(x)$")
plt.plot(x, logy1,'r--', label="taylor_1")
plt.plot(x, logy2,'g--', label="taylor_2")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()
# fig2.savefig('fig2.png')
```
![taylor5](https://user-images.githubusercontent.com/37234822/61225164-fcb31200-a75a-11e9-89c0-03d8b6ca352a.png)

<br/>
<br/>

----


### 3. <img src="https://latex.codecogs.com/svg.latex?f(x)=log(1+e^x)">
```python
x1 = np.linspace(-2, 2, 100)
f = np.log(1+np.exp(x1))
f1 = np.log(2) + 1/2*x1
f2 = np.log(2) + 1/2*x1 + 1/8*x1*x1
fig3 = plt.figure()
plt.title("$y = log(1+e^x)$")
plt.plot(x1, f, 'k', label='log(1+exp(x))')
plt.plot(x1, f1, 'r--', label='taylor_1')
plt.plot(x1, f2, 'g--', label='taylor_2')
plt.legend(loc=0)
plt.grid(True)
plt.show()
# fig3.savefig('fig3.png')
```

![taylor6](https://user-images.githubusercontent.com/37234822/61225166-fcb31200-a75a-11e9-9bcc-376d4570b0ec.png)
