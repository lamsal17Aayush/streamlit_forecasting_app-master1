sudo apt install streamlit
import streamlit as st 
import pandas as pd
import numpy as np 
from prophet import Prophet 
from prophet.diagonistics import performance_metrics , cross_validation
from prophet.plot import plot_cross_validation_metric
import base64


st.title('Automated Time Series Forecasting')




### step 1: Import CSV file 
df = st.file_uploader('Import the time series csv file here')


if df is not None:
    data=pd.read_csv(df)
    data['ds']=pd.to_datetime(data['ds'],errors='coerce')
    
    
    st.write(data)
    
    max_date = data['ds'].max()
    
    #st.write(max_date)
    
    
### Step 2: Select Forecast Horizon 
periods_input =st.number_input('How many periods would you like to forecast into the future?',
min_value = 1,max_value=365)

if df is not None:
    m = Prophet()
    m.fit(data)