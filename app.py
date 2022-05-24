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

def user_input_features():

    id_client = st.sidebar.selectbox('ID Client',(df["SK_ID_CURR"]))

    df_user = df[df["SK_ID_CURR"] == id_client]

    st.table(df_user)

input_df = user_input_features()

features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',

       'AMT_BALANCE',

       'MONTHS_BALANCE',

       'AMT_CREDIT_LIMIT_ACTUAL']

df_features = df[features]

st.subheader('Client Input features')

st.write(df)

load_model = pickle.load(open('C:/Users/BNP Leasing Solution/Documents/Projet Streamlit/final_model.sav', 'rb'))

prediction = load_model.predict(df)

prediction_proba = load_model.predict_proba(df)

#write the output:

model_labels = np.array(['Solvable','Non solvable'])

st.write(model_labels[prediction])

st.subheader('Prediction Probability')

st.write(prediction_proba)

