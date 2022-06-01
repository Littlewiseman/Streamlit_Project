import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
from lime import lime_text

st.write("""
# Brice Prediction App 
""")

df = pd.read_csv("C:/Users/BNP Leasing Solution/Documents/Projet Streamlit/data_dashboard.csv")
#print (df.head(5))

load_model = pickle.load(open('C:/Users/BNP Leasing Solution/Documents/Projet Streamlit/final_model.sav', 'rb'))

df = df.set_index("SK_ID_CURR")

#Entrée de l'ID Client

id_client = st.sidebar.selectbox('ID Client',(df.index))

#Feautures du client sélectionné

df_user = df[df.index == id_client]

st.subheader('Client Input Features')

st.table(df_user)

df_results = df.copy()

def probability(df):
    probas = load_model.predict_proba(df)
    probas = [proba[0] for proba in probas]
    return probas

df_results["Forest_PROBA"]=probability(df)
df_results["RESULT"]=load_model.predict(df)
df_results['Solvable'] = df_results["Forest_PROBA"]
df_results['Non Solvable']= 1-df_results["Forest_PROBA"]

df_results['RESULT'] = df_results['RESULT'].replace([0], 'SOLVABLE')
df_results['RESULT'] = df_results['RESULT'].replace([1], 'NON SOLVABLE')

#Resultat du client sélectionné

df_results_user = df_results[df_results.index == id_client]

df_results_user = df_results_user.drop(['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'AMT_BALANCE', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'FLAG_PHONE', 'EXT_SOURCE_2', 'FLAG_DOCUMENT_3', 
    'FLAG_DOCUMENT_7', 'AMT_DRAWINGS_CURRENT', 'Forest_PROBA'],axis=1)

st.subheader('Client Credit Application Results')

st.table(df_results_user)

#df_results['Solvable'].plot(kind='pie',fontsize=15)
#plt.title('Credit Score Result',fontsize=20)
#plt.show()


#10 best score

df_results_best = df_results.sort_values (by='Forest_PROBA', ascending=False)
df_results_best = df_results_best.drop(['Forest_PROBA', 'RESULT', 'Solvable', 'Non Solvable'], axis=1)

st.subheader('Features of Best Prediction Scores Among Clients & Selected Client Features')

st.table(df_results_best.head(10))
st.table(df_user)


#Lime Explainer

df_target = pd.read_csv("C:/Users/BNP Leasing Solution/Documents/Projet Streamlit/target_data_dashboard.csv")
df_target = df_target.set_index("SK_ID_CURR")

df_target = df_target.merge(df_results, how='inner', on='SK_ID_CURR')

X = df_target.iloc[:,1:18]
y = df.iloc[:,0]

#lime_explainer = LimeTabularExplainer(X.values,
                             #feature_names=X.columns,
                             #discretize_continuous=False)

explainer = lime.lime_tabular.LimeTabularExplainer(int(X.values) ,class_names=['Solvable', 'Non Solvable'], feature_names = X.columns,
                                                   kernel_width=3, verbose=False)

exp = lime_explainer.explain_instance(id_client.values,
                                rf.predict_proba,
                                num_features=10)

#exp = int(exp)

st.markdown(exp.as_html(), unsafe_allow_html=True)



#st.table(df_target.head(10))








features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'AMT_BALANCE', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'FLAG_PHONE', 'EXT_SOURCE_2', 'FLAG_DOCUMENT_3', 
    'FLAG_DOCUMENT_7', 'AMT_DRAWINGS_CURRENT']

df_features = df[features]

st.subheader('ALL Clients Input Features')

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

