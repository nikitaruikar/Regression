#Importing the libraries
import numpy as np
import pandas as pd
dataset=pd.read_csv('D:\\cognitior\\Basics of data science\\dataset\\50_Startups.csv')
dataset

x = dataset.iloc[:,:-1].values

y=dataset.iloc[:,4].values

#for converting categorical data into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

pd.DataFrame(x)

x = x[:,1:]

pd.DataFrame(x)

#for spliting data into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


from sklearn.tree import DecisionTreeRegressor
regressor  = DecisionTreeRegressor()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_train)
y_pred

y_test

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=5)
regressor.fit(x_train,y_train)
len(y_train)


y_pred = regressor.predict(x_test)

y_pred

y_test

r2_score(y_test,y_pred)


import seaborn as sns
sns.pairplot(dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']])

import statsmodels.formula.api as sm
regressor_ols = sm.OLS(y_train,x_train).fit()

regressor_ols.summary()
pd.DataFrame(x_test)

x_bk = x_train[:, 3:4]
pd.DataFrame(x_bk)
