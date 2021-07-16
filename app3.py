import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
#from sklearn.preprocessing import StandardScaler
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM
from plotly import graph_objs as go

from PIL import Image
image = Image.open('index1.jpg')
st.image(image)
st.write("This App predict Amazon Stock price")
st.set_option('deprecation.showfileUploaderEncoding', False)
file_upload = st.file_uploader("Upload csv file for predictions", type="csv")





st.title('Make sure the csv File is in the same format  as stocks.csv before uploading to avoid Error')

if file_upload is not None:
    data = pd.read_csv(file_upload)
    st.subheader("Raw data")
    st.write(data.head())
    st.write(data.tail())

    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock Open'))
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock Close'))

        fig.layout.update(title_text="Time series data and slider under the graph",xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()
    periods=365
    data_train=data[["Date","Adj Close"]]
    data_train=data.rename(columns={"Date": "ds", "Adj Close": "y"})
    m = Prophet()
    m.fit(data_train)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    st.subheader("Forcast data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.write("Forcast graph")
    fig=plot_plotly(m,forecast)
    st.plotly_chart(fig)

    st.write('Forcast plot_components')
    fig1=m.plot_components(forecast)
    st.write(fig1)
