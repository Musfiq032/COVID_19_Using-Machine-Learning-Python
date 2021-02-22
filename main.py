# Description : This program detects if someone hase caorna using machine learnign and python


import pandas as pd
from IPython.core import history
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt
import numpy as nm

# Title
st.write("""
# COVID-19 Detection
Detect if someone has corona virus using machine learning and python
""")

# adding the image

image = Image.open('corona_image.jpg')

st.image(image, caption='ML', use_column_width=True)

# get the data

data = pd.read_csv('corona_test.csv')

st.subheader('Data Information :')

st.dataframe(data)

st.write(data.describe())

chart = st.bar_chart(data)

# splitting data into independent 'X' and 'Y'

X = data.iloc[:, 0:9].values

Y = data.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

st.sidebar.write("Symptoms")


# Get the feature

def get_user_input():
    fever = st.sidebar.selectbox("Fever", ("Yes", "NO"))
    if fever == "Yes":
        fever = 1;
    else:
        fever = 0;

    tiredness = st.sidebar.selectbox("Tiredness", ("Yes", "NO"))

    if tiredness == "Yes":
        tiredness = 1;
    else:
        tiredness = 0;
    dryCough = st.sidebar.selectbox("Dry Cough", ("Yes", "NO"))
    if dryCough == "Yes":
        dryCough = 1;
    else:
        dryCough = 0;

    difficultyinBreathing = st.sidebar.selectbox("Difficulty In Breathing", ("Yes", "NO"))

    if difficultyinBreathing == "Yes":
        difficultyinBreathing = 1;
    else:
        difficultyinBreathing = 0;

    soreThroat = st.sidebar.selectbox("Sore Throat", ("Yes", "NO"))
    if soreThroat == "Yes":
        soreThroat = 1;
    else:
        soreThroat = 0;

    pains = st.sidebar.selectbox("Pain", ("Yes", "NO"))
    if pains == "Yes":
        pains = 1;
    else:
        pains = 0;
    nasal_congestion = st.sidebar.selectbox("Nasal Congestion", ("Yes", "NO"))
    if nasal_congestion == "Yes":
        nasal_congestion = 1;
    else:
        nasal_congestion = 0;
    runny_nose = st.sidebar.selectbox("Runny nose", ("Yes", "NO"))
    if runny_nose == "Yes":
        runny_nose = 1;
    else:
        runny_nose = 0;

    diarrhea = st.sidebar.selectbox("Diarrhea", ("Yes", "No"))
    if diarrhea == "Yes":
        diarrhea = 1;
    else:
        diarrhea = 0;

        user_data = {'fever': fever,
                     'tiredness': tiredness,
                     'drycough': dryCough,
                     'difficultyinBreathing': difficultyinBreathing,
                     'sorethroat': soreThroat,
                     'pain': pains,
                     'nasal_congestion': nasal_congestion,
                     'runny_nose': runny_nose,
                     'diarrhea': diarrhea,

                     }

        features = pd.DataFrame(user_data, index=[0])
        return features


# store the user input into variable

user_input = get_user_input()

st.subheader("User Input :")
st.write(user_input)

# create and train the model

RandomForestClassifier = RandomForestClassifier()
model =RandomForestClassifier.fit(X_train, Y_train)

# metrics

st.subheader('Model Test Accuracy: ')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

prediction = RandomForestClassifier.predict(user_input)

st.subheader('Model Train Accuracy: ')
st.write(str(accuracy_score(Y_train, RandomForestClassifier.predict(X_train))* 100)+ '%')



# Classification
st.subheader('Result: ')

if prediction== 1:
    st.write(str('You are Corona positive. Please Contact your nearby hospital for further inspection'))
else:
    st.write(str('You are Corona negative. Please Contact your nearby hospital for further inspection'))

