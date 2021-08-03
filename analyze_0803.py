# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 00:52:14 2021

@author: K
"""

import pandas as pd
import numpy as np

df=pd.read_csv('C:/py/pjt/report3_5.csv')
df.columns
df.info()

#등급컬럼 추가하기 #1-5단계
def grademaker(x):
    if x<4:
        return 1
    elif x>=4 and x<5:
        return 2
    elif x>=5 and x<6:
        return 3
    elif x>=6 and x<7:
        return 4
    else:
        return 5

df['Grade']=df['Life Ladder'].astype(int)
df['Grade']=df['Grade'].apply(lambda x:grademaker(x))
df['Grade'].unique()

#나라이름 원핫인코딩
one_c=pd.get_dummies(df['Country'])
one_c.head()

#합치기 
ndf=pd.concat([df,one_c],axis=1)

#속성, 독립변수 나누기 
x=ndf[ndf.columns.difference(['Country','Grade','Life Ladder','Year'])]
y=ndf['Grade']

#정규화
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
x=preprocessing.StandardScaler().fit(x).transform(x)
x_train,x_test,y_train,y_test=train_test_split\
    (x,y,test_size=0.2, random_state=4)
print(len(x_train))
print(len(x_test))

##KNN
from sklearn.neighbors import KNeighborsClassifier  
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train) 
y_pred=knn.predict(x_test)

#성능평가
from sklearn import metrics
knn_matrix=metrics.confusion_matrix(y_test,y_pred)
knn_matrix

#평가지표
knn_report=metrics.classification_report(y_test,y_pred)

#정확도, 정밀도, 재현율, F1 score
from sklearn.metrics import \
    accuracy_score, precision_score, recall_score,f1_score
print('정확도:',accuracy_score(y_test,y_pred))
print('정밀도:',precision_score(y_test,y_pred))
print('재현율:',recall_score(y_test,y_pred))
print('F1:',f1_score(y_test,y_pred))

###SVM
from sklearn import svm
svm_model=svm.SVC(kernel='rbf')
svm_model.fit(x_train,y_train)
y_pred=svm_model.predict(x_test)

#성능평가
from sklearn import metrics
svm_matrix=metrics.confusion_matrix(y_test,y_pred)
svm_matrix

#평가지표
svm_report=metrics.classification_report(y_test,y_pred)

#정확도, 정밀도, 재현율, F1 score
from sklearn.metrics import \
    accuracy_score, precision_score, recall_score,f1_score
print('정확도:',accuracy_score(y_test,y_pred))
print('정밀도:',precision_score(y_test,y_pred))
print('재현율:',recall_score(y_test,y_pred))
print('F1:',f1_score(y_test,y_pred))

###의사결정 나무 
from sklearn import tree
tree_model=tree.DecisionTreeClassifier(criterion='entropy',\
                                       max_depth=5)
tree_model.fit(x_train,y_train)
y_pred=tree_model.predict(x_test)

#성능평가
from sklearn import metrics
tree_matrix=metrics.confusion_matrix(y_test,y_pred)
tree_matrix

#평가지표
tree_report=metrics.classification_report(y_test,y_pred)

#정확도, 정밀도, 재현율, F1 score
from sklearn.metrics import \
    accuracy_score, precision_score, recall_score,f1_score
print('정확도:',accuracy_score(y_test,y_pred))
print('정밀도:',precision_score(y_test,y_pred))
print('재현율:',recall_score(y_test,y_pred))
print('F1:',f1_score(y_test,y_pred))

###GradientBoostingClassifier 
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

#성능평가
from sklearn import metrics
clf_matrix=metrics.confusion_matrix(y_test,y_pred)
clf_matrix

#평가지표
clf_report=metrics.classification_report(y_test,y_pred)

#정확도, 정밀도, 재현율, F1 score
from sklearn.metrics import \
    accuracy_score, precision_score, recall_score,f1_score
print('정확도:',accuracy_score(y_test,y_pred))
print('정밀도:',precision_score(y_test,y_pred))
print('재현율:',recall_score(y_test,y_pred))
print('F1:',f1_score(y_test,y_pred))

###Logistic 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=4)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

#성능평가
from sklearn import metrics
lr_matrix=metrics.confusion_matrix(y_test,y_pred)
lr_matrix

#평가지표
lr_report=metrics.classification_report(y_test,y_pred)

#정확도, 정밀도, 재현율, F1 score
from sklearn.metrics import \
    accuracy_score, precision_score, recall_score,f1_score
print('정확도:',accuracy_score(y_test,y_pred))
print('정밀도:',precision_score(y_test,y_pred))
print('재현율:',recall_score(y_test,y_pred))
print('F1:',f1_score(y_test,y_pred))






















