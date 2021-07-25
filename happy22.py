# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 00:54:22 2021

@author: mgk37
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Log GDP per capita                   36
Social support                       13
Healthy life expectancy at birth     55
Freedom to make life choices         32
Generosity                           89
Perceptions of corruption           110
Positive affect                      22
Negative affect                      16
dtype: int64
'''
worldh=pd.read_csv('world-happiness-report.csv')
worldh.info()
worldh.isnull().sum()
worldh['Life Ladder']
###
pm=worldh['Perceptions of corruption'].mean()
worldh['Perceptions of corruption'].max()
worldh['Perceptions of corruption'].min()
worldh['Perceptions of corruption'].value_counts().idxmax()
### 부정부패 인식 범위 -> 0.035~0.983, 평균 0.747, 결측값 110개
### 최빈값 0.844
gm=worldh['Log GDP per capita'].mean()
worldh['Log GDP per capita'].max()
worldh['Log GDP per capita'].min()
worldh['Log GDP per capita'].value_counts().idxmax()
### GDP범위 -> 6.635~11.638, 평균 9.368, 결측값 36개
### 0.844
sm=worldh['Social support'].mean()
worldh['Social support'].max()
worldh['Social support'].min()
worldh['Social support'].value_counts().idxmax()
### 사회적 도움범위 -> 0.29~0.987, 평균 0.812, 결측값 13개
### 0.818
hm=worldh['Healthy life expectancy at birth'].mean()
worldh['Healthy life expectancy at birth'].max()
worldh['Healthy life expectancy at birth'].min()
worldh['Healthy life expectancy at birth'].value_counts().idxmax()
### 건강 기대 범위 -> 32.3~77.1, 평균 63.359, 결측값 55개
### 72.2
gnm=worldh['Generosity'].mean()
worldh['Generosity'].max()
worldh['Generosity'].min()
worldh['Generosity'].value_counts().idxmax()
### 관대함 -> -0.335~0.698, 평균 0.0001, 결측값 89개
### -0.055
nm=worldh['Negative affect'].mean()
worldh['Negative affect'].max()
worldh['Negative affect'].min()
worldh['Negative affect'].value_counts().idxmax()
### 부정적인 영향 ->0.083~0.705, 평균 0.268, 결측값 16개
### 0.206
pam=worldh['Positive affect'].mean()
worldh['Positive affect'].max()
worldh['Positive affect'].min()
worldh['Positive affect'].value_counts().idxmax()
### 긍정적인영향 인식 범위 -> 0.322~0.944, 평균 0.71, 결측값 22개
### 0.784
fm=worldh['Freedom to make life choices'].mean()
worldh['Freedom to make life choices'].max()
worldh['Freedom to make life choices'].min()
worldh['Freedom to make life choices'].value_counts().idxmax()
### 긍정적인영향 인식 범위 -> 0.258~0.985, 평균 0.742, 결측값 32개
### 0.838
''' 
1.평균값
2.최빈값
3.중앙값
4.0 or 제거 
'''
##평균으로 채우기 
worldh1=pd.read_csv('world-happiness-report.csv')
worldh1.info()
worldh1['Log GDP per capita'].fillna(gm,inplace=True)
worldh1['Social support'].fillna(sm,inplace=True)
worldh1['Healthy life expectancy at birth'].fillna(hm,inplace=True)
worldh1['Freedom to make life choices'].fillna(fm,inplace=True)
worldh1['Generosity'].fillna(gnm,inplace=True)
worldh1['Perceptions of corruption'].fillna(pm,inplace=True)
worldh1['Positive affect'].fillna(pam,inplace=True)
worldh1['Negative affect'].fillna(nm,inplace=True)
worldh1.isnull().sum()
worldh1.columns
worldh1.info()
worldh1.tail()
del worldh1['Country name']
del worldh1['sjh']
worldh1['sjh']=np.nan

worldh1.sort_values(by='Life Ladder',inplace=True)


worldh1a=worldh1[worldh1['Life Ladder']<4]
worldh1b=worldh1[(worldh1['Life Ladder']>=4) & (worldh1['Life Ladder']<5)]
worldh1c=worldh1[(worldh1['Life Ladder']>=5) & (worldh1['Life Ladder']<6)]
worldh1d=worldh1[(worldh1['Life Ladder']>=6) & (worldh1['Life Ladder']<7)]
worldh1e=worldh1[worldh1['Life Ladder']>=7]

w1=worldh1a.copy()
w1['sjh']=1
w1

w2=worldh1b.copy() 
w2['sjh']=2
w2

w3=worldh1c.copy()
w3['sjh']=3
w3

w4=worldh1d.copy()
w4['sjh']=4
w4

w5=worldh1e.copy() 
w5['sjh']=5
w5


new_worldh1=pd.concat([w1,w2,w3,w4,w5],axis=0)
new_worldh1

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

#독립, 종속변수 선정 
x=new_worldh1[new_worldh1.columns.difference(['sjh'])]
y=new_worldh1['sjh']

#정규화
x=preprocessing.StandardScaler().fit(x).transform(x)

#훈련데이터, 검증데이터 분리 
x_train,x_test,y_train,y_test=train_test_split\
    (x,y,test_size=0.2, random_state=4)

#Logistic 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=4)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

print(y_test[:10].values) # [4 1 5 2 5 2 2 2 5 2]
print(y_pred[:10])        # [4 1 5 2 5 2 2 2 5 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 0.9076923076923077

#GBC
from sklearn.ensemble import GradientBoostingClassifier 
clf=GradientBoostingClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

print(y_test[:10].values) # [4 1 5 2 5 2 2 2 5 2]
print(y_pred[:10])        # [4 1 5 2 5 2 2 2 5 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 1.0

#svm 모델 
from sklearn import svm
svm_model=svm.SVC(kernel='rbf')
svm_model.fit(x_train,y_train)
y_pred=svm_model.predict(x_test)

print(y_test[:10].values) #[4 1 5 2 5 2 2 2 5 2]
print(y_pred[:10])        #[4 1 5 2 5 2 2 3 5 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 0.8692307692307693

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

print(y_test[:10].values) #[4 1 5 2 5 2 2 2 5 2]
print(y_pred[:10])        #[4 2 5 2 5 2 2 3 5 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 0.7410256410256411


#의사결정 트리 
from sklearn import tree
tree_model=tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
tree_model.fit(x_train,y_train)
y_pred=tree_model.predict(x_test)

print(y_test[:10].values) # [4 1 5 2 5 2 2 2 5 2]
print(y_pred[:10])        # [4 1 5 2 5 2 2 2 5 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
# 정확도(accirucy): 1.0

###결측행 전부 제거 
worldh2=pd.read_csv('world-happiness-report.csv')
worldh2.info()
worldh2=worldh2.dropna(axis=0)
del worldh2['Country name']

worldh2.sort_values(by='Life Ladder',inplace=True)


worldh2a=worldh2[worldh2['Life Ladder']<4]
worldh2b=worldh2[(worldh2['Life Ladder']>=4) & (worldh2['Life Ladder']<5)]
worldh2c=worldh2[(worldh2['Life Ladder']>=5) & (worldh2['Life Ladder']<6)]
worldh2d=worldh2[(worldh2['Life Ladder']>=6) & (worldh2['Life Ladder']<7)]
worldh2e=worldh2[worldh2['Life Ladder']>=7]

w1=worldh2a.copy()
w1['sjh']=1
w1

w2=worldh2b.copy() 
w2['sjh']=2
w2

w3=worldh2c.copy()
w3['sjh']=3
w3

w4=worldh2d.copy()
w4['sjh']=4
w4

w5=worldh2e.copy() 
w5['sjh']=5
w5


new_worldh2=pd.concat([w1,w2,w3,w4,w5],axis=0)
new_worldh2

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from math import sqrt
from sklearn.metrics import mean_squared_error

#독립, 종속변수 선정 
x=new_worldh2[new_worldh2.columns.difference(['sjh'])]
y=new_worldh2['sjh']

#정규화
x=preprocessing.StandardScaler().fit(x).transform(x)

#훈련데이터, 검증데이터 분리 
x_train,x_test,y_train,y_test=train_test_split\
    (x,y,test_size=0.2, random_state=4)

#Logistic 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=4)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

print(y_test[:10].values) # [2 3 3 3 3 2 3 3 3 1]
print(y_pred[:10])        # [2 3 3 3 3 2 4 3 4 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
# 정확도(accirucy): 0.9005847953216374

#GBC
from sklearn.ensemble import GradientBoostingClassifier 
clf=GradientBoostingClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

print(y_test[:10].values) # [2 3 3 3 3 2 3 3 3 1]
print(y_pred[:10])        # [2 3 3 3 3 2 3 3 3 1]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 0.9970760233918129

#svm 모델 
from sklearn import svm
svm_model=svm.SVC(kernel='rbf')
svm_model.fit(x_train,y_train)
y_pred=svm_model.predict(x_test)

print(y_test[:10].values) # [2 3 3 3 3 2 3 3 3 1]
print(y_pred[:10])        # [2 3 3 3 3 2 4 3 4 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 0.8538011695906432

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

print(y_test[:10].values) # [2 3 3 3 3 2 3 3 3 1]
print(y_pred[:10])        # [3 2 3 3 4 2 4 2 3 2]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 0.7485380116959064

#의사결정 트리 
from sklearn import tree
tree_model=tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
tree_model.fit(x_train,y_train)
y_pred=tree_model.predict(x_test)

print(y_test[:10].values) # [2 3 3 3 3 2 3 3 3 1]
print(y_pred[:10])        # [2 3 3 3 3 2 3 3 3 1]

print('정확도(accirucy):', accuracy_score(y_test,y_pred))
#정확도(accirucy): 0.9970760233918129
