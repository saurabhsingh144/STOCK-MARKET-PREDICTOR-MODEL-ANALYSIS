# -*- coding: utf-8 -*-
"""linear regression model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fFlH9HThtI-Bm3FdjVwi5IZf59EOevWU
"""

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sb

pip install YahooFinancials

pip install yfinance

tsla_df = yf.download('TSLA',
                      start='2012-01-01',
                      end='2022-05-10',
                      progress=False)

tsla_df.head()

tsla_df.tail()

tsla_df

tsla_df.info()

tsla_df.describe()

x = tsla_df[['High', 'Low', 'Open', 'Volume']].values
y = tsla_df['Close'].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
x_train

y_train

regressor = LinearRegression()

regressor.fit(x_train, y_train)

print(regressor.coef_)

print(regressor.intercept_)

predicted = regressor.predict(x_test)

print(predicted)

regressor.score(x_test,y_test)

df2 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted' : predicted.flatten()})

df2.head(100)

import math
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,predicted))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test,predicted)))

graph = df2.head(25)

plt.figure(figsize=(12,12))
graph.plot(kind ='bar')
plt.xlabel('Time')
plt.ylabel('price')



