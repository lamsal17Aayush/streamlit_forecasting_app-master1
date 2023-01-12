import streamlit as st 
import pandas as pd
import numpy as np 
from prophet import Prophet 
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
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
    
# Step 3: Visualize forecast data 
# The below visual shows future predicted values. "yhat" is the predicted 

if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    forecast = m.predict(future)
    fcst = forecast(['ds','yhat_lower','yhat_upper'])
    fcst_filtered= fcst[fcst['ds']>max_date]
    st.write(fcst_filtered)
    
    
    """ The next visual shows the actual (black dots) and predicted 
    """
    fig1 = m.plot(forecast)
    st.write(fig1)
    
    
    fig2 = m.plot_components(forecast)
    st.write(fig2)
""" 
if df is not None:
    csv_exp=fcst_filtered.to_csv(index=False)
    
    
    b64=base64.b64encode(csv_exp.encode()).decode() 
    href=f'<a href="data:file/csv;base64,{b64}">Download CSV file</a>  (right-click and save as ** &lt;forecast_name&ft;.csv***)'
    st.markdown(href,unsafe_allow_html=True)
"""  
    
    
    
    
    
    
    