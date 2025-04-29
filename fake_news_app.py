import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?.>+', "", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load your trained model
model = joblib.load('gradient_boosting_fakenews_model.pkl')
vectorizer = joblib.load('vectorizer_fakenews.pkl')

st.title("Fake News Detector")

# Input from user
user_input = st.text_area("Enter the news text:")

if st.button("Detect"):
    # user_input = wordopt(user_input)
    # transformed_text = vectorizer.transform([user_input])
    # prediction = model.predict(transformed_text)

    testing_news = {"text":[user_input]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred = model.predict(new_xv_test)
    


    if pred[0] == 1:
        st.success("This looks like REAL news!")
    else:
        st.error("This looks like FAKE news!")
