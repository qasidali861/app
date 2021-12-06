import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import plotly.express as px
import numpy as np

# Page layout
######################################
## Page expands to full width
st.set_page_config(page_title='Smart Feature Selection Web Application',
                   layout='wide')


######################################
## Page Title and sub title
######################################
st.title("Neural Lab Feature Selction App")
st.write("**Made By: Neural Lab**")
st.header("Machine Learning Classification Problem")
st.write("Optimal Feature Selection for any dataset in the  manner.")
st.write(" Fastest way to show the impact of our approach. All it needs is a small API.")
st.write("Just show the client the core output they want to see.")

######################################
## Sidebar
######################################
# Input your csv
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
# st.sidebar.markdown("""
# [Example CSV input file]
# (https://raw.githubusercontent.com/asad-mahmood/66DaysOfData/main/Heart%20Failure/heart_failure_clinical_records_dataset.csv)
# """)

######################################
# Main panel
######################################
st.subheader('Dataset')


######################################
# Feature Selection Code
######################################
def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Selection Plot',
                    width=1000, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)


def randomForest(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    st.subheader('Random Forest Classifier:')
    impPlot(feat_importances, 'Random Forest Classifier')
    #st.write(feat_importances)
    st.write('\n')


def extraTress(x, y):
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    st.subheader('Extra Trees Classifier:')
    impPlot(feat_importances, 'Extra Trees Classifier')
    st.write('\n')

def XGBoost(x, y):
    model = XGBClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    st.subheader('XGBoost Classifier:')
    impPlot(feat_importances, 'XGBoost Classifier')
    #st.write(feat_importances)
    st.write('\n')


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(5))
    x = df.iloc[:, :-1]  # Using all column except for the last column as X
    y = df.iloc[:, -1]  # Selecting the last column as Y
    randomForest(x, y)
    extraTress(x, y)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        df = pd.read_csv(
            'https://raw.githubusercontent.com/asad-mahmood/66DaysOfData/main/Heart%20Failure/heart_failure_clinical_records_dataset.csv')
        st.markdown('The **Heart Attack** dataset is used as the example.')
        st.write(df.head(5))
        x = df.iloc[:, :-1]  # Using all column except for the last column as X
        y = df.iloc[:, -1]  # Selecting the last column as Y
        randomForest(x, y)
        extraTress(x, y)
