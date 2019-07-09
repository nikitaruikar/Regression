#Predicting the balance by using the regression and used the diffrent techniques of regression

#importing the libraries and dataset
import numpy as np
import pandas as pd
import seaborn as sns
dataset=pd.read_excel('D:\\cognitior\\Basics of data science\\dataset\\credit.xlsx')
dataset

#Data Analysis and removing outliers in dataset
dataset = dataset[~((dataset<(Q1-1.5*IQR))
                       | (dataset>(Q3+1.5*IQR))).any(axis=1)]

dataset

Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3-Q1

x = dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
pd.DataFrame(y)
pd.DataFrame(x)


dataset1=dataset.drop(['Balance'],axis=1)
dataset1 = pd.get_dummies(dataset1, columns=['Gender','Ethnicity','Student','Married'], drop_first=True)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

 
x=dataset1.iloc[:,0:9]
y

#USing LInear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
regressor.coef_
regressor.intercept_
import statsmodels.formula.api as sm
regressor_ols = sm.OLS(y_train,x_train).fit()

regressor_ols.summary()

#Using random forest
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=5)
regressor.fit(x_train,y_train)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
# this are the veriables which are useful
dataset1

x=(dataset[['Income', 'Limit', 'Age', 'Education']])

