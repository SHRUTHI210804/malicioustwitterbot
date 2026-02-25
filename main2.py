import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(page_title="AI Twitter Bot Intelligence System", layout="wide")

st.title("🛡 AI-Powered Twitter Bot Intelligence System")
st.markdown("### Major Project - Behavioral Analysis Based Social Media Bot Detection")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📂 Upload Twitter Dataset (CSV)", type=["csv"])

if uploaded_file:

    dataset = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully")
    st.dataframe(dataset.head())

    # ===============================
    # FEATURE SELECTION
    # ===============================
    X = dataset[['followers_count', 'friends_count', 'listedcount',
                 'favourites_count', 'statuses_count', 'verified']]
    y = dataset['bot']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===============================
    # MODEL COMPARISON SECTION
    # ===============================
    if st.button("📊 Run Model Performance Analysis"):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Support Vector Machine": SVC(probability=True)
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred)
            recall = recall_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            auc = roc_auc_score(y_test, pred)

            st.markdown("---")
            st.subheader(f"🔎 {name} Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy*100:.2f}%")
            col2.metric("Precision", f"{precision*100:.2f}%")
            col3.metric("Recall", f"{recall*100:.2f}%")

            st.write(f"F1 Score: {f1:.2f}")
            st.write(f"AUC Score: {auc:.2f}")

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, pred)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="AUC = %0.2f" % auc)
            ax.plot([0, 1], [0, 1], "r--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - {name}")
            ax.legend(loc="lower right")
            st.pyplot(fig)

    # ===============================
    # FINAL MODEL FOR PREDICTION
    # ===============================
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_train, y_train)

    # ===============================
    # UNIQUE USERNAME BASED DETECTION
    # ===============================
    st.markdown("---")
    st.header("🔍 Social Media Intelligence Deep Scanner")

    username_input = st.text_input("Enter Twitter Username (exact match from dataset)")

    if st.button("🚀 Run Deep Scan"):

        dataset['screen_name_lower'] = dataset['screen_name'].str.lower()

        if username_input.lower() in dataset['screen_name_lower'].values:

            user_data = dataset[
                dataset['screen_name_lower'] == username_input.lower()
            ].iloc[0]

            input_features = np.array([[ 
                user_data['followers_count'],
                user_data['friends_count'],
                user_data['listedcount'],
                user_data['favourites_count'],
                user_data['statuses_count'],
                user_data['verified']
            ]])

            prediction = final_model.predict(input_features)
            probability = final_model.predict_proba(input_features)

            bot_prob = probability[0][1] * 100
            real_prob = probability[0][0] * 100

            st.markdown("---")
            st.subheader("🧠 AI Intelligence Report")

            col1, col2 = st.columns(2)
            col1.metric("Authenticity Score", f"{real_prob:.2f}%")
            col2.metric("Threat Probability", f"{bot_prob:.2f}%")

            st.progress(int(bot_prob))

            # Threat Classification
            if bot_prob > 75:
                st.error("🚨 THREAT LEVEL: HIGH")
                summary = "Severe automated bot-like behavioral patterns detected."
            elif bot_prob > 40:
                st.warning("⚠ THREAT LEVEL: MEDIUM")
                summary = "Moderate suspicious activity observed."
            else:
                st.success("✅ THREAT LEVEL: LOW")
                summary = "Behavior consistent with genuine human interaction."

            st.markdown("### 🤖 AI Summary")
            st.write(summary)

            st.markdown("### 📊 Behavioral Intelligence Card")

            ratio = user_data['followers_count'] / (user_data['friends_count'] + 1)

            st.write(f"Followers: {user_data['followers_count']}")
            st.write(f"Friends: {user_data['friends_count']}")
            st.write(f"Follower-Friend Ratio: {ratio:.2f}")
            st.write(f"Statuses Posted: {user_data['statuses_count']}")
            st.write(f"Listed Count: {user_data['listedcount']}")
            st.write(f"Verified: {user_data['verified']}")

            st.markdown("---")
            st.subheader("🎯 Final Classification")

            if prediction[0] == 1:
                st.error("CLASSIFIED AS: BOT ACCOUNT")
            else:
                st.success("CLASSIFIED AS: AUTHENTIC HUMAN ACCOUNT")

        else:
            st.warning("⚠ Username not found in dataset.")