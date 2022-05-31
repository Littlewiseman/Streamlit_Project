import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt

st.write("""
# Brice Prediction App 
""")

df = pd.read_csv("C:/Users/BNP Leasing Solution/Documents/Projet Streamlit/data_dashboard.csv")
#print (df.head(5))

load_model = pickle.load(open('C:/Users/BNP Leasing Solution/Documents/Projet Streamlit/final_model.sav', 'rb'))

df = df.set_index("SK_ID_CURR")

def user_input_features():

    id_client = st.sidebar.selectbox('ID Client',(df.index))
    df_user = df[df.index == id_client]
    st.table(df_user)

input_df = user_input_features()

df_results = df.copy()

df_results['prediction'] = load_model.predict(df)
#df_results['prediction_proba'] = load_model.predict_proba(df)

def user_input_results():

    id_client = st.sidebar.selectbox('ID Client',(df_results.index))
    df_user = df_results[df_results.index == id_client]
    st.table(df_user)

input_df = user_input_results()

features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',

       'AMT_BALANCE', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL']

df_features = df[features]

st.subheader('Client Input features')

st.write(df_features)

prediction = load_model.predict(df)
prediction_proba = load_model.predict_proba(df)

df_results = df.copy()
#print (df_results.head(5))

#df_results['prediction'] = load_model.predict(df)
#df_results['prediction_proba'] = load_model.predict_proba(df)

#results = ['prediction', 'prediction_proba']

#id_client = st.sidebar.selectbox('ID Client',(df.index))

#df_results_user = df_results[df_results.index == id_client]

#st.subheader('Client results')

#st.write(df_result_user)

#df_result_user = df_results[df_results.index == id_client]
#("prediction", "prediction_proba")

#write the output:

model_labels = np.array(['Solvable','Non solvable'])

st.write(model_labels[prediction])

st.subheader('Prediction Probability')

st.write(prediction_proba)

