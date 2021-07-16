import streamlit as st
from datetime import datetime
import yfinance as yt
import pandas as pd
#import pandas_datareader.data as web
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


from PIL import Image
image = Image.open('index1.jpg')
st.image(image)
start="2015-01-01"
today=datetime.today().strftime("%Y-%m-%d")

st.title("Stock prediction app,connecting to New York stock exchange(Live)")
stocks='MSFT'
selected_stock=st.selectbox("Select the dataset for processing",stocks)
n_years=st.slider("Years of prediction:how many years do you need for future?",1,5)
periods=n_years*365
@st.cache
def import_data(stock):
    data=pd.read_csv('AMZN.csv')

    data.reset_index(inplace=True)
    return data
data_import=st.text("load data...")
data=import_data(selected_stock)
data_import.text("loading data...done")
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
