import pandas as pd
import numpy
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import numpy as np
from keras.layers import LSTM, Dense, Dropout

from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import math

# pandas is imported to read the csv file and perform preprocessing on the dataset.
# matlpotlib is used to visualize the plot
# MinMaxScalar is used to normalize the value before training
from sklearn.preprocessing import MinMaxScaler
# numoy is used to deal with the data after train and split as data will be in form of aray for training and testing.
# keras has 2 models one is functional and another is sequential
from keras.models import Sequential
from keras.optimizers import SGD
from flask import Flask
import os


df = pd.read_csv('https://raw.githubusercontent.com/suhailbasha/Pridict/master/1minstock.csv', sep=",")
available_indicators = df['Ticker'].unique()

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash(name = __name__, server = server)
app.config.supress_callback_exceptions = True

app.layout = html.Div([
    html.H2('stock prediction'),
    dcc.Dropdown(
        id='x1',
        options=[{'label': i, 'value': i} for i in available_indicators],
        value='TATACOFFEE.NSE'

    ),
    dcc.Graph(
        id="g1",
        figure={'layout': {
            'height': 600,
            'width': 1500,
        }}
    ),
])

@app.callback(Output(component_id='g1',component_property='figure'),
              [Input(component_id='x1',component_property='value')],
              )
def update_fig(input_value):
    if input_value is not None:
        dff = df[df.Ticker.str.contains(input_value)]

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


        # The LSTM architecture
        regressorLG = Sequential()
        # First LSTM layer with Dropout regularisation
        regressorLG.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        regressorLG.add(Dropout(0.2))
        regressorLG.add(GRU(units=100, activation='tanh'))
        regressorLG.add(Dropout(0.2))
        # The output layer
        regressorLG.add(Dense(units=1))
        # Compiling the RNN
        regressorLG.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
        # Fitting to the training set
        regressorLG.fit(x_train, y_train, epochs=5, batch_size=32)

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
        LG_predicted_stock_price = regressorLG.predict(x_test)
        LG_predicted_stock_price = sc.inverse_transform(LG_predicted_stock_price)

        df1 = pd.DataFrame(y_test, columns=['a'])
        df2 = pd.DataFrame(LG_predicted_stock_price, columns=['a'])

        for i in range(len(df1)):
            if i % 1000 == 0:
                lstm1 = df1[i:i + 100]

        for i in range(len(df1)):
            if i % 1000 == 100:
                lstm2 = df1[i:i + 500]

        f1 = pd.DataFrame(lstm1, columns=['a'])

        trace1 = go.Scatter(
            x=dff['Time'],
            y=f1['a'],
            line=dict(color='green'),
            visible=True,
            name="actual price",
        )

        trace2 = go.Scatter(
            x=dff['Time'],
            y=df2["a"],
            line=dict(color='red'),
            visible=True,
            name="lstm predicted price",
            showlegend=True)
        fig = go.Figure(data=[trace1, trace2])

        return fig

