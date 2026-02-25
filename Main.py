# Streamlit version of Main.py as app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

words = ['bot', 'cannabis', 'tweet me', 'mishear', 'follow me', 'updates', 'every', 'gorilla', 'forget']

def getFrequency(bow):
    return sum(bow.get(word, 0) for word in words)

st.title("Detecting Malicious Twitter Bots Using Machine Learning")

uploaded_file = st.file_uploader("Upload Tweets Dataset", type=["csv"])

if uploaded_file:
    dataset = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")
    st.dataframe(dataset.head())

    if st.button("Run Module 1: Extract Tweets"):
        st.subheader("Extracted Tweets")
        st.write(dataset[['screen_name', 'status', 'name']].head())

    if st.button("Run Module 2: Recognize Twitter Bots using ML"):
        users = []
        for _, row in dataset.iterrows():
            if not row['verified']:
                bow = defaultdict(int)
                data = f"{row['screen_name']} {row['name']} {row['status']}"
                tokens = re.findall(r'\w+', data.lower())
                for token in tokens:
                    bow[token] += 1
                if getFrequency(bow) > 0 and row['listedcount'] < 16000 and row['followers_count'] < 200:
                    users.append(row['screen_name'])

        st.write("Possible BOT users:", users)

        X = dataset[['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
        y = dataset['bot']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression().fit(X_train, y_train)
        pred = model.predict(X_test)

        st.write("Accuracy: {:.2f}%".format(accuracy_score(y_test, pred) * 100))
        st.write("Precision: {:.2f}%".format(precision_score(y_test, pred) * 100))
        st.write("Recall: {:.2f}%".format(recall_score(y_test, pred) * 100))
        st.write("F1 Score: {:.2f}".format(f1_score(y_test, pred)))
        st.write("AUC: {:.2f}".format(roc_auc_score(y_test, pred)))

        fpr, tpr, _ = roc_curve(y_test, pred)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc_score(y_test, pred))
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Twitter Bots')
        ax.legend(loc='lower right')
        st.pyplot(fig)

    if st.button("Run Module 3: Recognize Malicious URLS using ML"):
        dataset['URLS'] = dataset['status'].apply(lambda x: 1 if 'http' in str(x) else 0)
        X = dataset[['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified', 'URLS']]
        y = dataset['bot']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression().fit(X_train, y_train)
        pred = model.predict(X_test)

        st.write("Accuracy: {:.2f}%".format(accuracy_score(y_test, pred) * 100))
        st.write("Precision: {:.2f}%".format(precision_score(y_test, pred) * 100))
        st.write("Recall: {:.2f}%".format(recall_score(y_test, pred) * 100))
        st.write("F1 Score: {:.2f}".format(f1_score(y_test, pred)))
        st.write("AUC: {:.2f}".format(roc_auc_score(y_test, pred)))

        fpr, tpr, _ = roc_curve(y_test, pred)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc_score(y_test, pred))
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Malicious URLs')
        ax.legend(loc='lower right')
        st.pyplot(fig)
