import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

st.title('👾 Machine Learning App 🤖')

st.info('This is app builds a machine learning model!')
with st.expander('Data'):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write("**X-Feature**")
  X_raw = df.drop('species',axis=1)
  X_raw
  
  st.write("**y-Target**")
  y_raw = df['species']
  y_raw
with st.expander('Data Visualization'):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

with st.sidebar:
  st.header("Input Feature.")
  island = st.selectbox('Island',('Biscoe','Torgersen','Dream'))
  bill_length_mm = st.slider("Bill length (mm)",32.1,59.6,43.9)
  bill_depth_mm = st.slider("Bill Depth (mm)",13.1,21.5,17.2)
  flipper_length_mm = st.slider("flipper length (mm)",172.0,231.0,201.0)
  body_mass_g = st.slider("Body mass (g)",2700.0,6300.0,4207.0)
  gender = st.selectbox('Gender',('male','female'))
  with st.spinner("Loading..."):
    time.sleep(0.5)
  st.success("Done!")

  data = {'island' : island,
         'bill_length_mm' : bill_length_mm,
         'bill_depth_mm' : bill_depth_mm,
          'flipper_length_mm' : flipper_length_mm,
          'body_mass_g' : body_mass_g,
          'sex' : gender
         }
  input_df = pd.DataFrame(data,index=[0])
  input_penguins = pd.concat([input_df,X_raw],axis=0)
with st.expander('Input Features'):
  st.write("**Input Selected**")
  input_df
  
#encode X
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
X = df_penguins[1:]
input_row = df_penguins[:1]

#encode Y
target_mapper = {'Adelie' : 0,
                'Chinstrap': 1,
                'Gentoo':2}
def target_enocde(val):
  return target_mapper[val]
y = y_raw.apply(target_enocde)

with st.expander('Data Preparation'):
  st.write("**Encoded X**")
  input_row
  st.write("**Encoded Y**")
  y

#Model Selection

clf = RandomForestClassifier()
clf.fit(X,y)

#Apply Model to make prediction
prediction = clf.predict(input_row)
prediction_prob = clf.predict_proba(input_row)
df_pred_prob = pd.DataFrame(prediction_prob)
df_pred_prob.columns = ['Adelie','Chinstrap','Gentoo']

st.subheader("**Predicted Species**")
st.dataframe(df_pred_prob,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)
pen_sp = np.array(['Adelie','Chinstrap','Gentoo'])
st.success(str(pen_sp[prediction][0]))
