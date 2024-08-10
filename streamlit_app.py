import streamlit as st
import pandas as pd
import time

st.title('👾 Machine Learning App 🤖')

st.info('This is app builds a machine learning model!')
with st.expander('Data'):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write("**X-Feature**")
  X = df.drop('species',axis=1)
  X
  
  st.write("**y-Target**")
  y = df['species']
  y
with st.expander('Data Visualization'):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

with st.sidebar:
  st.header("Important Feature.")
  island = st.selectbox('Island',('Biscoe','Torgersen','Dream'))
  gender = st.selectbox('Gender',('Male','Female'))
  bill_length_mm = st.slider("Bill length (mm)",32.1,59.6,43.9)
  with st.spinner("Loading..."):
    time.sleep(5)
  st.success("Done!")
