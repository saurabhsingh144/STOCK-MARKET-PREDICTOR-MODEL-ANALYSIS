# STOCK-MARKET-ANALYSIS-USING MODELS
Stock Market Analysis Using Different Machine Learning Approach includes LSTM/Linear Regression/SVM
Stock Price Prediction
Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange. The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy. 
            
    Project implementation
A. Database Acquisition 
1. Columns Open and Close - the starting and final price at which the  
stock is traded on a particular day. 
2. High and Low represent the maximum, minimum, and last price of the  
share for the day. 
3. Volume is the number of shares bought or sold in the day. 
![Screenshot 2023-11-10 123153](https://github.com/saurabhsingh144/STOCK-MARKET-PREDICTOR-MODEL-ANALYSIS/assets/88964851/058ed121-9cb6-4b73-88c0-bcc840c9b8dd)

B. Data Preprocessor

1.Sorted the dataset in ascending order then created a separate dataset. Dropped
specific values/columns also any new feature created does not affect the original
data.

2.Loaded the dataset, defined the target variable for the problem.

       TECHNIQUES DEVELOPED / METHODOLOGY AND ADAPTIVE ALGORITHMS
   
The set of algorithms used for developing this Project:

• Relative Strength Index 

• Linear Regression

• Long Short -Term Memory (LSTM)


Relative Strength Index
   
The Relative Strength Index (RSI) is a measurement used by traders to assess the
price momentum of a stock or other security. The basic idea behind the RSI is to
measure how quickly traders are bidding the price of the security up or down. The
RSI plots this result on a scale of 0 to 100. Readings below 30 generally indicate
that the stock is oversold, while readings above 70 indicate that it is overbought.
Traders will often place this RSI chart below the price chart for the security, so they
can compare its recent momentum against its market price.
RSI Formula
=100−[100/1+ Average Gain/Average Loss]
The average gain or loss used in the calculation is the average percentage
gain or loss during a look-back period. The formula uses a positive value
for the average loss. Periods with price losses are counted as 0 in the
calculations of average gain, and periods when the price increases are
counted as 0 for the calculation of average losses.
The standard is to use 14 periods to calculate the initial RSI value. For
example, imagine the market closed higher seven out of the past 14 days
with an average gain of 1%. The remaining seven days all closed lower
with an average loss of −0.8%.

FLOWCHART 

![Uptrend_Strategy_Blueprint](https://github.com/saurabhsingh144/STOCK-MARKET-PREDICTOR-MODEL-ANALYSIS/assets/88964851/d98c2fb8-b2be-4141-866a-b6822f087175)


C. Steps For Algorithm

    Step 1: Closing Price - We will take the closing price of the stock for 30 days.

    Step 2: Changes in Closing Price
    We then compare the closing price of the current day with the previous day’s
    closing price and note them down.

    Step 3: Gain and Loss
    We will now create two sections depending on the fact the price increased or
    decreased, with respect to the previous day’s closing price.
    If the price has increased, we note down the difference in the “Gain” column
    and if it's a loss, then we note it down in the “Loss” column.

    Step 4: Average Gain and Loss
    In the RSI indicator, to smoothen the price movement, we take an average of the gains (and
    losses) for a certain period.
    While we call it an average, a little explanation would be needed. For the first 14 periods, it is a
    simple average of the values.

    Step 5: Calculate RS
    Now, to make matters simple, we add a column called “RS” which is simply, (Avg
    Gain)/(Avg Loss).

    Step 6: Calculation of RSI
    RSI = [100 - (100/{1+ RS})].
    determining information to be employed in the current neural networks. For immediate
    For example, for (14-05),
    RSI = [100 - (100/{1+ RS})] = [100 - (100/{1+ 1.24})] = 55.37.
    In this manner, the table is updated.
    This is how we get the value of RSI. The RSI indicator graph is always created with respect to the
    closing price.
		
		
	 
Linear Regression
	
Linear Regression is a machine learning algorithm based on supervised learning. It
performs a regression task. Regression models a target prediction value based on
independent variables. It is mostly used for finding out the relationship between variables
and forecasting. Different regression models differ based on – the kind of relationship
between dependent and independent variables they are considering, and the number of
independent variables getting used.
Linear regression performs the task to predict a dependent variable value (y) based on
a given independent variable (x). So, this regression technique finds out a linear
relationship between x (input) and y(output). Hence, the name is Linear Regression. In
the figure above, X (input) is the work experience and Y (output) is the salary of a
person. The regression line is the best fit line for our model.

D. Algorithm for Linear Regression / Steps Involved

    1. import some required libraries
    2. import matplotlib.pyplot as plt
    3. import pandas as pd
    4. import numpy as np
    5. Define the dataset
    6. x= np.array([2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4])
    7. y= np.array([2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8])
    8. n= np.size(x)
    9. Plot the data points
    10. plt.scatter(experience,salary, color = 'red')
    11. plt.xlabel("Experience")
    12. plt.ylabel("Salary")
    13. plt.show()

FLOWCHART

![1_HJnsZhHTY0hVJH7qA1FiDg](https://github.com/saurabhsingh144/STOCK-MARKET-PREDICTOR-MODEL-ANALYSIS/assets/88964851/c994310e-55c9-43d7-b411-6f3b0a46b79a)

LSTM Model

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN)
architecture used in the field of deep learning (DL). Unlike standard feedforward neural
networks, LSTM has feedback connections. It can process not only single data points
(such as images), but also entire sequences of data (such as speech or video). For
example, LSTM is applicable to tasks such as unsegmented, connected handwriting
recognition,[2] speech recognition and anomaly detection in network traffic or IDSs
(intrusion detection systems).
A common LSTM unit is composed of a cell, an input gate, an output gate and a forget
gate. The cell remembers values over arbitrary time intervals and the three gates
regulate the flow of information into and out of the cell.
LSTM networks are well-suited to classifying, processing and making predictions based
on time series data, since there can be lags of unknown duration between important
events in a time series. LSTMs were developed to deal with the vanishing gradient
problem that can be encountered when training traditional RNNs. Relative insensitivity
to gap length is an advantage of LSTM over RNNs, hidden Markov models and other
sequence learning methods in numerous applications.

D. Steps to implement LSTM / Algorithm

    1. Let’s load the data and inspect them:
    2. import math
    3. import matplotlib.pyplot as plt
    4. import keras
    5. import pandas as pd
    6. import numpy as np
    7. from keras.models import Sequential
    8. from keras.layers import Dense
    9. from keras.layers import LSTM
    10. from keras.layers import Dropout
    11. from keras.layers import *
    12. from sklearn.preprocessing import MinMaxScaler
    13. from sklearn.metrics import mean_squared_error
    14. from sklearn.metrics import mean_absolute_error
    15. from sklearn.model_selection import train_test_split
    16. from keras.callbacks import EarlyStopping
    17. df=pd.read_csv("TSLA.csv")
    18. print(‘Number of rows and columns:’, df.shape)
    19. df.head(5)
    20. The next step is to split the data into training and test sets to avoid overfitting and to
    be able to investigate the generalization ability of our model.
    The target value to be predicted is going to be the “Close” stock price
    value. training_set = df.iloc[:800, 1:2].values
    test_set = df.iloc[800:, 1:2].values
    21. Now, it’s time to build the model. We will build the LSTM with 50 neurons and 4 hidden
    layers. Finally, we will assign 1 neuron in the output layer for predicting the normalized stock
    price. We will use the MSE loss function and the Adam stochastic gradient descent
    optimizer.
    22. model = Sequential()
    23. Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape =
    (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    24. Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    25. Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    26. Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    27. Adding the output layer
    model.add(Dense(units = 1))
    28. Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    29. Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    The Final Fitting is seen like this as shown

FLOWCHART

![pone 0222365 g002](https://github.com/saurabhsingh144/STOCK-MARKET-PREDICTOR-MODEL-ANALYSIS/assets/88964851/4f0b2579-1bbe-4442-ab59-bd074b85801d)

F.  RESULT (OUTPUT)

Predictive modeling significantly reduces the cost required for companies to
forecast business outcomes, environmental factors, competitive intelligence, and
market conditions The proposed system is trained and tested over the dataset taken
from various companies. It is split into training and testing sets respectively and
yields the results upon passing through the different models.


![Screenshot 2023-11-10 131552](https://github.com/saurabhsingh144/STOCK-MARKET-PREDICTOR-MODEL-ANALYSIS/assets/88964851/11ad980c-85c0-4e24-bf52-473db12b8cc8)


   
