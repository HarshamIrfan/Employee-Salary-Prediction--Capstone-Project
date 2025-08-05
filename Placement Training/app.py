#app.py

import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(page_title='Salary Estimation App',layout='wide')
st.title('Salary Estimation App' )
st.markdown('##### Predict your expected salary based on company experience!')

st.image('https://cdn.dribbble.com/userupload/21275225/file/original-25999973ddba8d4037fdcdae521317ce.gif',caption='Let us predict!',use_container_width=True)

st.divider()

col1,col2,col3=st.columns(3)

with col1:
    years_at_company= st.number_input=('Years at company',min_value=0,max_value=20,value=3)

with col2:
    satisfaction_level=st.slider('Satisfaction Level',min_value=0.0 ,max_value=1.0, step=0.1,value=0.7)

with col3:
    average_monthly_hours=st.slider('Average Monthly Hours',min_value=120,max_value=310,step=1,value=160)

scaler=joblib.load('scaler.pkl')
model=joblib.load('model.pkl')

predict_button=st.button('Predict Salary')
st.divider()

if predict_button:
    st.balloons()

    x_array=scaler.transform([np.array(x)])
    prediction=model.predict(x_array_array)

    st.success(f'Predicted Salary: Rs. {prediction[0]:,.2f}')

    df_inout=pd.DataFrame({
        'Feature':['Years at company','Satisfaction Level','Average Monthly Hours'],
        'Value':x
    })

    fig = px.bar(df_input,x="Feature",y="Value",color="Feature",title="Your Input Profile",text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆Enter details and press the **Predict Salary** button.")
    

