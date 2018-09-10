import pandas as pd
import numpy as np
# https://github.com/tensorflow/tensorflow/issues/2375
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from time import sleep


# Data Preprocessing
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Fitting simple linear regression model to training set
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


# Predecting the result
Y_pred = regressor.predict(X_test)


# Visualization
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()


plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()