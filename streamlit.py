import streamlit as st
import pandas as pd
import seaborn as sns
from joblib import load
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance, plot_partial_dependence

@st.cache(allow_output_mutation=True)
def load_model():
    print('loading model')
    return load('neural_bialetti.joblib')


@st.cache()
def load_data():
    print('loading data')
    df = pd.read_csv('Export Spedire - db_bialetti (1).csv')
    return df


model = load_model()
df = load_data()
df=df.fillna( method='ffill')


st.title('Welcome to Cassandra')


user_input= {}




categorical = ['day_week', 'month','ios_14']

for feat in categorical:
    unique_values = df[feat].unique()
    user_input[feat]=st.sidebar.selectbox(feat, unique_values)

numerical = ['fb_spent','google_spent','organico','cambio_penusd']

for feat in numerical:
    v_min = float(df[feat].min())
    v_max =float(df[feat].max())
    user_input[feat]=st.sidebar.slider(
        feat,
        min_value= v_min,
        max_value=v_max,
        value= (v_min+v_max)/2
    )
X = pd.DataFrame([user_input])
st.write(X)



z=(X['fb_spent']+X['google_spent'])

cpa= z/model.predict(X)

prediction =model.predict(X)

st.title('Previsione transazioni')


fig= go.Figure(
    go.Indicator(
        mode= 'gauge+number',
        value=prediction[0]
    )
)

st.plotly_chart(fig)
st.write(prediction)
fig_cpa= go.Figure(
    go.Indicator(
        mode= 'number',
        value= cpa[0]
    )
)
st.title('CPA')
st.plotly_chart(fig_cpa)

aov= 60

fatturato=model.predict(X)*aov

profit=fatturato*0.6-z


fig_profit= go.Figure(
    go.Indicator(
        mode= 'number',
        value= profit[0]
    )
)
st.subheader('Profit')
st.plotly_chart(fig_profit)

plot_partial_dependence(model, df, features=['google_spent'])
plot_partial_dependence(model, df, features=['fb_spent'])
plot_partial_dependence(model, df, features=['organico'])
plot_partial_dependence(model, df, features=['cambio_penusd'])
