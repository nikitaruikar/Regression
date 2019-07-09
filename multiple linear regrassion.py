#Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('D:\\cognitior\\Basics of data science\\dataset\\50_Startups.csv')

#This code for dependent and independent veriable
x = dataset.iloc[:,0:4].values

y = dataset.iloc[:,4].values


#For Encoding of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
onehotendoer = OneHotEncoder(categorical_features=[3])
x = onehotendoer.fit_transform(x).toarray()

pd.DataFrame(x)

x = x[:,1:]
pd.DataFrame(x)
#spliting the data into test and trains for module 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#Module 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the y 
y_pred = regressor.predict(x_test)
y_test

y_pred
#checking out accuracy of model
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
regressor.coef_
r2_score(y_test,y_pred)

regressor.intercept_
