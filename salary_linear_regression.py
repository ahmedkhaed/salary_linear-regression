# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:44:58 2019

@author: Ahmed Khaled

steps for linear regression are :
    step 1 :import libararies
    step 2 : Get data set
    step 3 : check missing data 
    step 4: check categeorical data
    step 5 : split data into input & output 
    step 6 : visulize your data to choice the best way(model) for this data
    step 7 :split data into training data  & test data 
    step 8 :Build your model
    step 9 :plot best line 
    step 10 :Estimate Error 
"""
#feature scalling 
"""from sklearn.preprocessing import StandardScaler
sc_x =StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test )"""


#Data processing
#step 1 :import libararies
import numpy as np   
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split



#step 2 : importing the data set 

path='C:\\Users\\Ahmed Khaled\\Downloads\\my work (regression)\\3)salary_linear_regression\\Salary_Data.csv' 
dataset  = pd.read_csv(path)
dataset.plot(kind='scatter',x='YearsExperience', y='Salary',color = 'red',figsize=(5,5))
print('data : \n ',dataset)
print('data.head : \n ',dataset.head())
print('data.describe : \n ',dataset.describe())
print('data.shape : \n ',dataset.shape)


#step 3 : check missing data 
# there are no missing data 



#step 4: check categeorical data
# there are no categeorical data




#step 5 : split data into input & output 
x =dataset["YearsExperience"].values.reshape(-1,1) #by header
y =dataset["Salary"].values.reshape(-1,1)

# OR you can use those but you have to convert to matrix
#x = dataset.iloc[:,0]
#y = dataset.iloc[:,1]

# OR you can use those here  you don't have to convert to matrix 
#x = dataset.iloc[:, :-1].values
#y = dataset.iloc[: , 1].values

#step 6 : visulize your data to choice the best way(model) for this data
plt.plot(x,y,'o',color='red')
plt.show()

#step 7 :split data into training data  & test data 
#x_train ,y_train = x[:20],y[:20] 
#x_test ,y_test = x[20:] ,y[20:] # note 1-D

#OR you can use  train_test_split function from sklearn this is  better 
#from sklearn.cross_validation import train_test_split
x_train, x_test ,y_train,y_test = train_test_split(x,y,test_size = 1/3 ,random_state = 0 )

#step 8 :Build your model,Fitting Simple Linear regression to the Training set
model = LinearRegression()   # model from library 
model.fit(x_train,y_train)   # for training

#step 9 :plot best line 
regression_line = model.predict(x)
plt.plot(x,regression_line,color= 'green')
plt.plot(x_train,y_train,'o',color = 'red')
plt.plot(x_test,y_test,'o',color = 'blue')
plt.show()

print('#####################################')

#visulize the traning set results 
plt.scatter(x_train , y_train, color ='red')
plt.plot(x_train  ,  model.predict(x_train), color = 'blue' )
plt.title('salary vs exprience (Traning set)')
plt.ylabel('salary')
plt.xlabel('Years of Exprience')
plt.show()

#visulize the traning set results 

plt.scatter(x_test , y_test, color ='red')
plt.plot(x_train  ,  model.predict(x_train), color = 'blue' )
plt.title('salary vs exprience (test set)')
plt.ylabel('salary')
plt.xlabel('Years of Exprience')
plt.show() 

print(model.predict(8))
#step 10 :Estimate Error 
y_pred = model.predict(x_test) #predicting the test set results
print('MSE = \n',mean_squared_error(y_test,y_pred))


#visulize the traning set results 
plt.scatter(x_train , y_train, color ='red')
plt.plot(x_train  ,  model.predict(x_train), color = 'blue' )
plt.title('salary vs exprience (Traning set)')
plt.ylabel('salary')
plt.xlabel('Years of Exprience')
plt.show()
#visulize the traning set results 
plt.scatter(x_test , y_test, color ='red')
plt.plot(x_train  ,  model.predict(x_train), color = 'blue' )
plt.title('salary vs exprience (test set)')
plt.ylabel('salary')
plt.xlabel('Years of Exprience')
plt.show() 