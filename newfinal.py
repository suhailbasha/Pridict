import pandas as pd
import numpy
import plotly.graph_objs as go
import numpy as np
from keras.layers import LSTM, Dense, Dropout
import plotly




from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import math





# pandas is imported to read the csv file and perform preprocessing on the dataset.
import pandas
# matlpotlib is used to visualize the plot
# MinMaxScalar is used to normalize the value before training
from sklearn.preprocessing import MinMaxScaler
# numoy is used to deal with the data after train and split as data will be in form of aray for training and testing.
# keras has 2 models one is functional and another is sequential
from keras.models import Sequential
from keras.optimizers import SGD

# Dense layer is the output layer
# LSTM is Long Term Short Term Memory










df = pd.read_csv('https://raw.githubusercontent.com/suhailbasha/Pridict/master/1minstock.csv')





dff=df[df['Ticker']=='TATACOFFEE.NSE']





training_set = dff.iloc[:, 4:5].values
test_set = dff.iloc[:, 6:7].values

sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)
x_train = []
y_train = []
timestamp = 60
length = len(training_set)
for i in range(timestamp, length):
    x_train.append(training_set_scaled[i - timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=92, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=92, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=92, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=92, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5, batch_size=3268)
test_set = dff

test_set = test_set.loc[:, test_set.columns == 'Close']

y_test = test_set.iloc[timestamp:, 0:].values

closing_price = test_set.iloc[:, 0:].values
closing_price_scaled = sc.transform(closing_price)
x_test = []
length = len(test_set)

for i in range(timestamp, length):
    x_test.append(closing_price_scaled[i - timestamp:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_pred = model.predict(x_test)
predicted_price = sc.inverse_transform(y_pred)

df1 = pd.DataFrame(y_test, columns=['a'])
df2 = pd.DataFrame(predicted_price, columns=['a'])

for i in range(len(df1)):
    if i % 1000 == 0:
        lstm1 = df1[i:i + 100]

for i in range(len(df1)):
    if i % 1000 == 100:
        lstm2 = df1[i:i + 500]

f1 = pd.DataFrame(lstm1, columns=['a'])


trace1 = go.Scatter(
    x=dff['Time'],
    y=df2['a'],
    line=dict(color='green'),
    visible=True,
    name="actual price",
)

trace2 = go.Scatter(
    x=dff['Time'],
    y=f1['a'],
    line=dict(color='red'),
    visible=True,
    name="lstm predicted price",
    showlegend=True)

trace3 = go.Scatter(
    x=dff['Time'],
    y=df1['a'],
    line=dict(color='red'),
    visible=True,
    name="lstm predicted price",
    showlegend=True)

data=[trace1, trace2,trace3]

fig=dict(data=data)
plotly.offline.plot(fig,filename='drop.html')



