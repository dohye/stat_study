나이브 베이즈 알고리즘을 이용한 movie_reviews의 neg / pos 분류하기
---
181020_statistics_class

<br/>

- 총 2000개의 영화 리뷰들을 바탕으로, neg/pos 분류기를 생성함  

- movie_reviews data : python 내장 corpus로 , 각 리뷰는 neg / pos 로 분류되어 있음  

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;N_{pos}=1000">, <img src="https://latex.codecogs.com/svg.latex?\Large&space;N_{neg}=1000">

<br/>

* 리뷰(에 있는 단어가) positive일 확률 : 
<img src="https://latex.codecogs.com/svg.latex?P(pos|words)=P(words|pos)"> <img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?P(pos)/P(words)">

* 리뷰(에 있는 단어가) negative일 확률 : 
<img src="https://latex.codecogs.com/svg.latex?P(neg|words)=P(words|neg)"> <img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?P(neg)/P(words)">

<br/>

* <img src="https://latex.codecogs.com/svg.latex?P(pos|words)"> 와 <img src="https://latex.codecogs.com/svg.latex?P(neg|words)"> 를 비교해 더 큰쪽으로 분류한다.  

* 여기서 <img src="https://latex.codecogs.com/svg.latex?P(pos)=1000/2000=1/2">, <img src="https://latex.codecogs.com/svg.latex?P(neg)=1000/2000=1/2">  이고  

* <img src="https://latex.codecogs.com/svg.latex?P(words)"> 는 대소 비교 시 생략가능 하므로, 결과적으로 <img src="https://latex.codecogs.com/svg.latex?P(words|pos)">와 <img src="https://latex.codecogs.com/svg.latex?(words|neg)">를 비교함    

<br/>

예를 들어,  

<img src="https://latex.codecogs.com/svg.latex?P(w_1,w_2,\cdot\cdot\cdot,w_n|pos)=P(w_1|pos)"> <img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?P(w_2|pos)">
<img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?\cdot\cdot\cdot"> <img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?P(w_n|pos)">

<img src="https://latex.codecogs.com/svg.latex?P(w_1,w_2,\cdot\cdot\cdot,w_n|neg)=P(w_1|neg)"> <img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?P(w_2|neg)">
<img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?\cdot\cdot\cdot"> <img src="https://latex.codecogs.com/svg.latex?\times"> <img src="https://latex.codecogs.com/svg.latex?P(w_n|neg)">

(각 단어는 독립임을 가정함)

<br/>
<br/>
<br/>

## Naive Bayes Classifier 코드


```python
from nltk import regexp_tokenize
from nltk.corpus import movie_reviews, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import numpy as np
```

<br/>

### 데이터의 형태
```python
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
```

![naive](https://user-images.githubusercontent.com/37234822/60964731-0e597b80-a34f-11e9-8668-27db0ad1725e.png)

<br/>

### nltk에 내장되어 있는 movie_review corpus 불러들이기
```python
file = movie_reviews.fileids()
neg_docs_id = list(filter(lambda doc : doc.startswith("neg"), file))
pos_docs_id = list(filter(lambda doc : doc.startswith("pos"), file))
neg_docs = [movie_reviews.raw(doc_id) for doc_id in neg_docs_id]
pos_docs = [movie_reviews.raw(doc_id) for doc_id in pos_docs_id]
```

* neg_docs  
![naive2](https://user-images.githubusercontent.com/37234822/60964778-2e893a80-a34f-11e9-932b-22e97fa36686.png)

* pos_docs  
![naive3](https://user-images.githubusercontent.com/37234822/60964779-2f21d100-a34f-11e9-8231-988a0eacf568.png)


<br/>

### split train/test

```python
neg_train = neg_docs[:750]
pos_train = pos_docs[:750]
neg_test = neg_docs[750:]
pos_test = pos_docs[750:]
```

<br/>

### 토크나이저 함수 정의

```python
def tokenizer(document):
    pattern = r"""[a-zA-Z]+"""
    stop_words = set(stopwords.words('english'))
    lowered_document = document.lower()
    regularized_token = regexp_tokenize(lowered_document, pattern)
    words = [word for word in regularized_token if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return words
```

<br/>

### negative/positive 문서별로 단어들 합치기

```python
neg = list()
for i in range(len(neg_train)):
    neg.append(tokenizer(neg_train[i]))
    
pos = list()
for i in range(len(pos_train)):
    pos.append(tokenizer(pos_train[i]))
def flatten(words):
    result  = []
    for i in words:
        if(isinstance(i, list)):
            result += flatten(i)
        else:
            result.append(i)
    return result
    
neg_words = flatten(neg)
pos_words = flatten(pos)
```

<br/>

### 단어의 확률 딕셔너리 만들기

```python
## 단어의 빈도수 
neg_count = Counter(neg_words)
pos_count = Counter(pos_words)  

## 제일 많이 등장한 5000개의 단어 딕셔너리
neg_most = dict(neg_count.most_common(5000))
pos_most = dict(pos_count.most_common(5000))  

## 단어별로 문서에 등장하는 확률 딕셔너리
neg_dict = dict()
for key in neg_most:
    neg_dict[key] = neg_most[key]/5000
pos_dict = dict()
for key in pos_most:
    pos_dict[key] = pos_most[key]/5000
```

<br/>

### neg_dict/ pos_dict의 형태

* neg_dict  
![naive4](https://user-images.githubusercontent.com/37234822/60964847-5aa4bb80-a34f-11e9-9c92-38d0d4f61c48.png)


* pos_dict  
![naive6](https://user-images.githubusercontent.com/37234822/60964848-5aa4bb80-a34f-11e9-8129-245aa0426d7a.png)

<br/>

### naive bayes classifier 함수 정의
```python
def nb_classifier(test, label):
    
    anss = list()
    
    for sent in test:
        
        sent = tokenizer(sent)
        pos_prob = 1
        neg_prob = 1
        
        for i in range(len(sent)):  
         
            pos_prob += np.log(pos_dict.get(sent[i],1/5000))        
            neg_prob += np.log(neg_dict.get(sent[i],1/5000))
            sub_prob = pos_prob - neg_prob
    
        if sub_prob > 0:
            ans = "positive"
        if sub_prob < 0:
            ans = "negative"
        if sub_prob == 0:
            ans = "same"
            
        anss.append(ans)
    
    anss = flatten(anss)
    ans_freq = Counter(anss)
    
    if label == "positive":
        accuracy = ans_freq["positive"]/len(test)
    elif label == "negative":
        accuracy = ans_freq["negative"]/len(test)
        
    return anss, accuracy
```


### 결과
```python
neg_res, neg_acc = nb_classifier(neg_test, label="negative")
pos_res, pos_acc = nb_classifier(pos_test, label="positive")
```

```python
print(neg_acc)
# negative accuracy : 0.496
print(pos_acc)
# positive accuracy : 0.944
```

<br/>
<br/>

그런데 여기서, negative인데 positive로 분류할 확률이 높다. 따라서 이 loss를 줄이기 위해 가중치를 조정해 보았다.

### naive bayes classifier 함수 재정의
```python
def nb_classifier(test, label):
    
    anss = list()
    lr = list()
    for sent in test:
        
        lrt = 0
        
        sent = tokenizer(sent)
        
        for i in range(len(sent)):  
         
            lrt += np.log((pos_dict.get(sent[i],1/5000)) / (neg_dict.get(sent[i],1/5000)))
        
        lr.append(lrt)
        if lrt > np.exp(3.32):
            ans = "positive"
        if lrt < np.exp(3.32):   # negative인데 positive로 분류할 loss를 줄이기 위한 가중치 조정
            ans = "negative"
        if lrt == np.exp(3.32):
            ans = "same"
            
        anss.append(ans)
    
    anss = flatten(anss)
    ans_freq = Counter(anss)
    
    if label == "positive":
        accuracy = ans_freq["positive"]/len(test)
    elif label == "negative":
        accuracy = ans_freq["negative"]/len(test)  
        
    return lr, anss, accuracy
```


### 결과
```python
_, neg_res, neg_acc = nb_classifier(neg_test, label="negative")
_, pos_res, pos_acc = nb_classifier(pos_test, label="positive")
```

```python
print(neg_acc)
# negative accuracy : 0.816
print(pos_acc)
# positive accuracy : 0.74
```

accuracy가 조정된 것을 확인할 수 있다.
