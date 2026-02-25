import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Account Investigation Console", layout="wide")

st.title("🛰️ AI Social Media Investigation Console")
st.markdown("### Intelligent Behavioral Analysis & Bot Risk Detection System")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully")
    st.dataframe(df.head())

    # Auto detect numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("### 🔧 Automatic Feature Detection")
    st.write("Detected Numeric Behavioral Features:")
    st.write(numeric_columns)

    # Choose target column
    target_column = st.selectbox("Select Target Column (Real/Fake or 0/1)", df.columns)

    # Choose username column
    username_column = st.selectbox("Select Username Column", df.columns)

    if st.button("🚀 Initialize AI Investigation System"):

        X = df[numeric_columns].fillna(0)
        y = pd.to_numeric(df[target_column], errors='coerce').fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        st.success("AI Investigation Engine Ready")
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        # Store in session
        st.session_state["model"] = model
        st.session_state["df"] = df
        st.session_state["features"] = numeric_columns
        st.session_state["username_col"] = username_column

# Investigation Section
if "model" in st.session_state:

    st.markdown("---")
    st.header("🔍 Run Account Investigation")

    username_input = st.text_input("Enter Username to Investigate")

    if st.button("Start Investigation"):

        df = st.session_state["df"]
        model = st.session_state["model"]
        features = st.session_state["features"]
        username_col = st.session_state["username_col"]

        df[username_col] = df[username_col].astype(str)

        if username_input in df[username_col].values:

            user = df[df[username_col] == username_input].iloc[0]

            input_data = np.array([user[features]])
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)

            bot_risk = probability[0][1] * 100
            real_score = probability[0][0] * 100

            st.markdown("## 🧾 Profile Snapshot")
            st.write(user)

            st.markdown("## ⚖ AI Verdict Scale")
            fig, ax = plt.subplots()
            ax.barh(["Real Score", "Bot Risk"], [real_score, bot_risk])
            ax.set_xlim(0, 100)
            st.pyplot(fig)

            st.markdown("## 🧠 AI Behavioral Explanation")

            high_features = user[features].sort_values(ascending=False).head(3)
            st.write("Top Influential Behavioral Indicators:")
            st.write(high_features)

            st.markdown("## 🎯 Final AI Verdict")

            if prediction[0] == 1:
                st.error("🚨 BOT / FAKE ACCOUNT DETECTED")
            else:
                st.success("✅ AUTHENTIC HUMAN ACCOUNT")

        else:
            st.warning("Username not found in dataset.")