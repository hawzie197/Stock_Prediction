import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


style.use('ggplot')             # specifying the type of plotting chart to use
df = quandl.get('WIKI/GOOGL')   # add stock to right of WIKI



# Adding columns to stock value charts
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)




X = np.array(df.drop(['label'], 1))                           # finding out X and Y values
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace=True)
Y = np.array(df['label'])
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)


clf = LinearRegression(n_jobs=-1)                             # figuring out the linear regression within the data set
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)                          # getting the confidence of the linear regression
forecast_set = clf.predict(X_lately)                          # predicting the future prices


print(df.head())                                              # prints the prices of the stock when it first came out
print()
print('--------------------------------------------')
print()
print(df.tail())                                              # prints the most recent prices for the stock
print()
print('--------------------------------------------')
print()
print('Predicted Prices over next ', forecast_out, ' days')   # Displays timeline of predicted prices
print()
print(forecast_set)                                           # Displays the future prices
print()
print('--------------------------------------------')
print()
print('Accuracy: ', accuracy)                                 # Displays the confidence of the linear regression

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400                                               # number of seconds in one day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]     # loops through columns to get numbers to graph



# plots and create the graph with all numbers
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=10)   # specifies the location of the key
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()
