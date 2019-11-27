import pandas as pd
import numpy
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import numpy as np
from google.cloud import storage 
from keras.layers import LSTM, Dense, Dropout

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

colors = {
    'background': '#111111',
    'text': 'white'
}

df = pd.read_csv('gs://grace_bucket/suhail/1minstock.csv', sep=",")

available_indicators = df['Ticker'].unique()

app = dash.Dash(__name__)

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


@app.callback(Output('g1', 'figure'),
              [Input('x1', 'value')],
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

        df1 = pd.DataFrame(y_test,columns=['a'])
        df2 = pd.DataFrame(predicted_price,columns=['a'])


        for i in range(len(df1)):
            if i % 1000 == 0:
             lstm1 = df1[i:i + 100]


        for i in range(len(df1)):
            if i % 1000 == 100:
                lstm2 = df1[i:i + 500]

        f1 = pd.DataFrame(lstm1,columns=['a'])

        # ann
        model = Sequential()
        # First GRU layer with Dropout regularisation
        model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))
        # Second GRU layer
        model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))
        # Third GRU layer
        model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))
        # Fourth GRU layer
        model.add(GRU(units=50, activation='tanh'))
        model.add(Dropout(0.2))
        # The output layer
        model.add(Dense(units=1))

        model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')

        # fitting the model

        model.fit(x_train, y_train, epochs=5, batch_size=32)

        predicted_with_gru = model.predict(x_test)
        predicted_with_gru = sc.inverse_transform(predicted_with_gru)

        df3 = pd.DataFrame(predicted_with_gru,columns=['a'])

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

        # Preparing X_test and predicting the prices

        LG_predicted_stock_price = regressorLG.predict(x_test)
        LG_predicted_stock_price = sc.inverse_transform(LG_predicted_stock_price)

        df4 = pd.DataFrame(LG_predicted_stock_price,columns=['a'])
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

        trace3 = go.Scatter(

            y=df3["a"],
            x=dff['Time'],
            line=dict(color='blue'),
            visible=True,
            name="gru prediction price",
            showlegend=True)

        trace4 = go.Scatter(
            x=dff['Time'],
            y=df1['a'],
            line=dict(color='green'),
            visible=True,
            name="actual price",
            showlegend=True)

        trace5 = go.Scatter(
            x=dff['Time'],
            y=df4["a"],
            line=dict(color='purple'),
            visible=True,
            name="regressor predicted price",
            showlegend=True)

        high = (f1) + 20
        low = df1["a"].tolist()
        close = df2["a"].tolist()
        close1 = df3["a"].tolist()
        close2 = df4["a"].tolist()

        frames = [dict(data=[dict(type='scatter',
                                  y=high[:k + 1]),
                             dict(type='scatter',
                                  y=close[:k + 1]),
                             dict(type='scatter',
                                  y=close1[:k + 1]),
                             dict(type='scatter',
                                  y=close2[:k + 1]),

                             dict(type='scatter',
                                  y=low[:k + 1])],
                       traces=[3, 1, 2, 3, 4],
                       # this means that  frames[k]['data'][0]  updates trace1, and   frames[k]['data'][1], trace2
                       ) for k in range(1, len(low) - 1)]

        layout = go.Layout(width=1500,
                           height=600,
                           showlegend=True,
                           hovermode='closest',
                           updatemenus=[dict(type='buttons', showactive=False,
                                             y=1.05,
                                             x=1.15,
                                             xanchor='right',
                                             yanchor='top',
                                             pad=dict(t=0, r=10),
                                             buttons=[dict(label='Play',
                                                           method='animate',
                                                           args=[None,
                                                                 dict(frame=dict(duration=3,
                                                                                 redraw=True),
                                                                      transition=dict(duration=5),
                                                                      fromcurrent=True,
                                                                      mode='immediate')])])])

        layout.update(yaxis=dict(range=[min(low) - 0.5, high.max() + 0.5], autorange=True));

        fig = go.Figure(data=[trace1, trace2, trace3, trace5, trace4], frames=frames, layout=layout)
        for template in ["plotly_dark"]:
            fig.update_layout(template=template)

        return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
