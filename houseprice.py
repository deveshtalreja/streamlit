import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

hide_st_style = ''' <style> #MainMenu {visibility: hidden;} footer{visibility: hidden;} </style>'''
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("# Dubai Real Estate Prediction")
st.markdown('## This app predicts the **House Prices in Dubai**')
st.write('---')
st.write('''
**Credits**
- App built by __Devesh Talreja__
- Built in `Python` using `streamlit`,`sklearn`, `pandas` and `numpy`
''')
st.write('---')


def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		bar.progress(i + 1)
		time.sleep(0.01)
		st.empty()
    
@st.cache
def load_data():
	df=pd.read_csv(r'C:\Users\USER\PYprojects\RealEststreamlit\properties_data.csv')
	df=df[df['price']>0]
	#df['floors']=df['floors'].astype(int)
	#df=df[df['no_of_bedrooms']>0]
	df=df[df['no_of_bathrooms']>0]
	return df


df=load_data()

#df
#de=df.describe()
#de=df.describe()=false, open=true, bold_text_alphanum=true if


st.sidebar.subheader('Property Options')
# Sidebar Options:
params={
'bedrooms' : st.sidebar.selectbox('Bedrooms',(0,1,2,3,4,5)),
'bathrooms' : st.sidebar.selectbox('Bathrooms',(1,2,3,4,5,6)),
'sqft' : st.sidebar.slider('Square Feet', min(df['size_in_sqft']),max(df['size_in_sqft']),step=100),
'balcony':1 if st.sidebar.checkbox('Balcony') else 0,
'furnished':0 if st.sidebar.checkbox('Furnished') else 1
}



def map_df(df):
	df=df[df['no_of_bedrooms']==params['bedrooms']]
	df=df[df['no_of_bathrooms']==params['bathrooms']]
	
	df=df[df['balcony']==params['balcony']]
	df=df[df['unfurnished']==params['furnished']]
	df=df[(df['size_in_sqft']>0.9*params['sqft']) & (df['size_in_sqft']<1.1*params['sqft'])]
	df.reset_index(drop=True)
	return df

test_size=st.sidebar.slider('Pick Test Size', 0.05,0.5,0.25,step=0.05)

@st.cache
def get_models():
	y=df['price']
	X=df[['no_of_bedrooms','no_of_bathrooms','size_in_sqft','balcony','unfurnished']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
	models = [DummyRegressor(strategy='mean'),	
			RandomForestRegressor(n_estimators=170,max_depth=25),
			DecisionTreeRegressor(max_depth=25),
		    GradientBoostingRegressor(learning_rate=0.01,n_estimators=200,max_depth=5), 
			LinearRegression(n_jobs=10, normalize=True)]
	df_models = pd.DataFrame()
	temp = {}
	print(X_test)
	#run through models
	for model in models:
		print(model)
		m = str(model)
		temp['Model'] = m[:m.index('(')]
		model.fit(X_train, y_train)
		temp['RMSE_Price'] = sqrt(mse(y_test, model.predict(X_test)))
		temp['Pred Value']=model.predict(pd.DataFrame(params,  index=[0]))[0]
		print('RMSE score',temp['RMSE_Price'])
		df_models = df_models.append([temp])
	df_models.set_index('Model', inplace=True)
	pred_value=df_models['Pred Value'].iloc[[df_models['RMSE_Price'].argmin()]].values.astype(float)
	return pred_value, df_models

def run_data():
	run_status()
	df_models=get_models()[0][0]
	st.write('## Given your parameters, the predicted value is **{:,.2f} AED**'.format(df_models))
	df1=map_df(df)
	st.map(df1)
	df1

def show_ML():
	df_models=get_models()[1]
	df_models
	st.write('## **This diagram shows root mean square error for all models used**')
	st.bar_chart(df_models['RMSE_Price'])

btn = st.sidebar.button("Predict")
if btn:
	run_data()
else:
	pass

if st.sidebar.checkbox('Show ML Models'):
	run_data()
	
	st.write('---')
	st.markdown('## **ML Models**')
	df_models=get_models()[1]
	df_models
	st.write('## **This diagram shows root mean sq error for all models**')
	st.bar_chart(df_models['RMSE_Price'])

if st.sidebar.checkbox('Show Raw Data'):
	st.write('---')
	st.markdown('## **Raw Data**')
	df

