# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:54:53 2020

@author: Abhinash
"""

Machine Learning Class 
Batch will start 2:00 pm

Supervised: data is Labeled

Linear Regression:
    what case study
    fundamental apply it
    model
    model evalution metric
    DVT(model performance)
    
    Regression: LR|MLR|LASSO|DTR(Decision Tree Regressor)
    |SVR(support vector Regressor)
    Data: Continuous

Classification:
    problem statement(Data Labeled):Binary|classes
        
case : weather it will rain today: event: YES|NO

case: plant species classifiy: plant:(3,4,5,6,,7)
class more than 2 multi class classification
        
classification: a)Logistic Regression
                b) Support Vector Machine:
            classification   support vector classifier
            Regression       support vector Regressor
                     
               c) Decision Tree:
                    class Decision Tree Classifier:
                    reg   Decision Tree Regressor
               d) KNN- K Nearest Neighbors
               e) Naive Bayes(text classification)
                  Text classification
               Ensemble technique
              
CASE STUDY: Predicting the survival rate in Titanic Disaster
#source: Kaggle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\Abhinash\Desktop\data.csv")
data.head()

#data explore:
#data shape
data.shape
#row: 891 col: 12

#data information for data
data.columns
data.info()

#stastical summary
data.describe()

#raw raw : clean
#percenatage of missing value in each col
per=data.isnull().sum()/len(data)
per
#data missing>70 col drop
#rest col fill with CTL

data.pop("Cabin")
data.isnull().sum()

#age fill distribution(continous form)
data["Age"].plot.hist()
data["Age"].mean()
data["Age"].median()
# 28.0 median mean 28.12 : UDF
# median :28 mena: 29.699 RS

data["Age"].fillna(data["Age"].median(),
             inplace=True)

#max people board ship from which location
data["Embarked"].value_counts()
plt.bar(np.array(["S","C","Q"]),
        np.array([644,168,77]),
        color=['r','g','b'])
plt.show()
#mode 

data["Embarked"].fillna("S",inplace=True)

#--------------------------------------
#drop all irrelevant col
#PassengerId|Name|Ticket
data=data.drop(["PassengerId",
                "Name","Ticket"],axis=1)
data.info()
#---------------------------------------

#text data(object type)->numeric form convert
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    if data[i].dtype=='O':
        data[i]=le.fit_transform(data[i])
data.info()


#label encoder
#gender
'''Male:1
Female:0
'''
data["Sex"].head()
'''Embarked
S:2
C:0
Q:1
'''
data["Embarked"].head()
#matrix form dummy varaible

#sep dep and indep
#dependent: Survived(yes|no)
y=data["Survived"]
x=data.drop(['Survived'],axis=1)


#feature Selection:
#regression: OLS
#classficication:

import statsmodels.api as sm
X_con=sm.add_constant(x)

model=sm.Logit(y,X_con).fit()
model.summary()

x1=X_con.drop(["Parch"],axis=1)
model1=sm.Logit(y,x1).fit()
model1.summary()

x2=x1.drop(['Fare'],axis=1)
model2=sm.Logit(y,x2).fit()
model2.summary()

#logit equation
'''Y(survived)=5.4256-1.14(pclass)-2.715(Sex)
-0.038(age)-0.3356(sibsp)-0.2356(embarked)'''

X=x.loc[0:,['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked']]
X.head()
#loc= columns name : name
#iloc= columns index value

#VIF:
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculating VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif
#NO-Multicollinearity

#file: training
#model select: prediction: binary
# Classification: LogisticRegression

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#model train on train data
model.fit(X,y)

#data prepare for test file
data=pd.read_csv("test.csv")
data.head()

#data explore:
#data shape
data.shape
#row: 891 col: 12

#data information for data
data.columns
data.info()

#stastical summary
data.describe()

#raw raw : clean
#percenatage of missing value in each col
per=data.isnull().sum()/len(data)
per
#data missing>70 col drop
#rest col fill with CTL

data.pop("Cabin")
data.isnull().sum()

#age fill distribution(continous form)
data["Age"].plot.hist()
data["Age"].mean()
data["Age"].median()
# 28.0 median mean 28.12 : UDF
# median :28 mena: 29.699 RS

data["Age"].fillna(data["Age"].median(),
             inplace=True)

#max people board ship from which location
data["Embarked"].value_counts()
plt.bar(np.array(["S","C","Q"]),
        np.array([644,168,77]),
        color=['r','g','b'])
plt.show()
#mode 

data["Embarked"].fillna("S",inplace=True)

#--------------------------------------
#drop all irrelevant col
#PassengerId|Name|Ticket
data=data.drop(["PassengerId",
                "Name","Ticket"],axis=1)
data.info()
#---------------------------------------

#text data(object type)->numeric form convert
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    if data[i].dtype=='O':
        data[i]=le.fit_transform(data[i])
data.info()

test=data.iloc[0:,[0,1,2,3,6]]
test.head()

#model test
pred=model.predict(test)
print(pred)

sub=pd.DataFrame({"PasID":data["PassengerId"],
                 "Survived":pred})
sub.head()
#dataframe csv file convert
sub.to_csv('submison.csv')

#-----------------------------------------------
#model accuarcy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\Abhinash\Desktop\data.csv")
data.head()

#data explore:
#data shape
data.shape
#row: 891 col: 12

#data information for data
data.columns
data.info()

#stastical summary
data.describe()

#raw raw : clean
#percenatage of missing value in each col
per=data.isnull().sum()/len(data)
per
#data missing>70 col drop
#rest col fill with CTL

data.pop("Cabin")
data.isnull().sum()

#age fill distribution(continous form)
data["Age"].plot.hist()
data["Age"].mean()
data["Age"].median()
# 28.0 median mean 28.12 : UDF
# median :28 mena: 29.699 RS

data["Age"].fillna(data["Age"].median(),
             inplace=True)

#max people board ship from which location
data["Embarked"].value_counts()
plt.bar(np.array(["S","C","Q"]),
        np.array([644,168,77]),
        color=['r','g','b'])
plt.show()
#mode 

data["Embarked"].fillna("S",inplace=True)

#--------------------------------------
#drop all irrelevant col
#PassengerId|Name|Ticket
data=data.drop(["PassengerId",
                "Name","Ticket"],axis=1)
data.info()
#---------------------------------------

#text data(object type)->numeric form convert
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    if data[i].dtype=='O':
        data[i]=le.fit_transform(data[i])
data.info()


#label encoder
#gender
'''Male:1
Female:0
'''
data["Sex"].head()
'''Embarked
S:2
C:0
Q:1
'''
data["Embarked"].head()
#matrix form dummy varaible

#sep dep and indep
#dependent: Survived(yes|no)
y=data["Survived"]
x=data.drop(['Survived'],axis=1)


#feature Selection:
#regression: OLS
#classficication:

import statsmodels.api as sm
X_con=sm.add_constant(x)

model=sm.Logit(y,X_con).fit()
model.summary()

x1=X_con.drop(["Parch"],axis=1)
model1=sm.Logit(y,x1).fit()
model1.summary()

x2=x1.drop(['Fare'],axis=1)
model2=sm.Logit(y,x2).fit()
model2.summary()

#logit equation
'''Y(survived)=5.4256-1.14(pclass)-2.715(Sex)
-0.038(age)-0.3356(sibsp)-0.2356(embarked)'''

X=x.loc[0:,['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked']]
X.head()
#loc= columns name : name
#iloc= columns index value

#VIF:
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculating VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif
#NO-Multicollinearity

#file: training
#model select: prediction: binary
# Classification: LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30)



from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#model train on train data
model.fit(x_train,y_train)

pred=model.predict(x_test)

#accuarcy of model
from sklearn.metrics import accuracy_score
print("Model Accuracy is :",accuracy_score(y_test,pred))

#---------------------------------------------
#THEORY : LOGISTIC | METRICS EVALTE MODEL
CONFUSION MATRIX
ROC
AUC
K-FOLD





























