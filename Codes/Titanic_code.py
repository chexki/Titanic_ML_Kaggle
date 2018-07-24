# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:14:00 2018

@author: Chexki
"""

# Titanic Mine

import pandas as pd
import numpy as np

#%%

ttrain = pd.read_csv(r'D:\DATA SCIENCE\Python\Project\Titanic\train.csv')
ttest = pd.read_csv(r'D:\DATA SCIENCE\Python\Project\Titanic\test.csv')

print(ttrain.head())

#%%
print(ttrain.tail())
print(ttrain.dtypes)
print(ttrain.shape)

#%%
# Descriptive statistics

print(ttrain.describe())
print(ttrain.describe(include="all"))

print(ttrain['Pclass'].value_counts())

# Cross Tabulation
print(pd.crosstab(ttrain['Sex'],ttrain['Pclass']))

# Sex wise Survival rate
print(ttrain['Sex'][ttrain['Survived']== 1].value_counts())

#%%
# Feature Selction
training_data = ttrain[['Pclass','Sex','Age','Fare','Survived']]

#%%
# Missig

print(training_data.isnull().sum())
print(ttest.isnull().sum())

## Handling missing values in Age Column

training_data['Age'].fillna(training_data.Age.mean(),inplace=True)
ttest['Age'].fillna(ttest.Age.mean(),inplace=True)
ttest['Fare'].fillna(ttest.Fare.median(),inplace=True)
print(training_data.isnull().sum())

#%%
PassengerAge=training_data['Age']
PassengerAge=PassengerAge.dropna()
Bins= [0,15,21,61,PassengerAge.max()+1]
Binlabels=['Children','Adolescents','Adult','Senior']
categories= pd.cut(PassengerAge, Bins, labels=Binlabels, right=False,
                   include_lowest= True)
print(categories.value_counts())
print(categories)

#%%
colname =['Pclass', 'Sex']
# Preprocessing
from sklearn import preprocessing
le={}           
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    training_data[x]= le[x].fit_transform(training_data.__getattr__(x))
    ttest[x]= le[x].fit_transform(ttest.__getattr__(x))
print(training_data.head())

#%%
X = training_data.values[:,:-1]          #independent vars
Y = training_data.values[:,-1]           # dependent var
Y=Y.astype(int)

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

#%%
# Training the Model   (X-Train, Y_train)
from sklearn.model_selection import train_test_split
#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,
                                                    random_state=10)

#%%
from sklearn.linear_model import LogisticRegression
# create a model
classifier = (LogisticRegression())
classifier.fit(X_train,Y_train)

acc_classifier = round(classifier.score(X_train, Y_train) * 100, 2)
acc_classifier

Y_pred = classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

print(Y_test)
#%%

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm= confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification Report")
print(classification_report(Y_test,Y_pred))

accuracy_score = accuracy_score(Y_test,Y_pred)
print("Accuracy of the Model:", accuracy_score)

#%%
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)
#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.65:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)


from sklearn.metrics import confusion_matrix, accuracy_score
cfm= confusion_matrix(Y_test.tolist(),y_pred_class)
print(cfm)

accuracy_score = accuracy_score(Y_test.tolist(),y_pred_class)
print("Accuracy of the Model:", accuracy_score)
#%%
# Cross Validation
classifier=(LogisticRegression())
from sklearn import cross_validation

# Performing kfold cross validation
kfold_cv = cross_validation.KFold(n=len(X_train),n_folds = 10)
print(kfold_cv)

# running the model using scoring metric accuracy

kfold_cv_result = cross_validation.cross_val_score( \
                                        estimator =classifier,
                                        X=X_train,y=Y_train,
                                        scoring="accuracy",
                                        cv=kfold_cv)
print(kfold_cv_result)

# finding the mean
print(kfold_cv_result.mean())
#%%
for train_value, test_value in kfold_cv:
    classifier.fit(X_train[train_value],Y_train[train_value]). \
    predict(X_train[test_value])
    
Y_pred = classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm= confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification Report")
print(classification_report(Y_test,Y_pred))
accuracy_score = accuracy_score(Y_test,Y_pred)
print("Accuracy of the Model:", accuracy_score)   

#############################################################################################
#%%
## Preparing test data for prediction

ttest1 = ttest[['Pclass','Sex','Age','Fare']]

X1 = ttest1.values[:,:]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)
print(X1)


#%%
# Predictin Logistic Regression

output = classifier.predict(X1).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = ttest['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('D:/DATA SCIENCE/Python/Project/Titanic/Chetan_Titanic.csv', index=False)

#######################################################################################
#%%

#predicting using the KNeighbors_Classifier
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=2, metric='euclidean')

#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)
Y_pred = model_KNN.predict(X_test)

#%%

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
from sklearn.metrics import accuracy_score
for K in range(15):
    Kvalue = K+1
    model_KNN = KNeighborsClassifier(Kvalue)
    model_KNN.fit(X_train, Y_train) 
    Y_pred = model_KNN.predict(X_test)
    print ("Accuracy is ", accuracy_score(Y_test,Y_pred), "for K-Value:",Kvalue)
    
#%%
output = model_KNN.predict(X1).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = ttest['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('D:/DATA SCIENCE/Python/Project/Titanic/Chetan_Titanic_knn.csv', index=False)

#%%
#Predicting using Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
model_RandomForestClassifier=(RandomForestClassifier(501))

#fit the model on data and predict the values
model_RandomForestClassifier.fit(X_train,Y_train)
Y_pred = model_RandomForestClassifier.predict(X_test)
#%%
accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
output = model_RandomForestClassifier.predict(X1).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = ttest['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('D:/DATA SCIENCE/Python/Project/Titanic/Chetan_Titanic_rf.csv', index=False)

