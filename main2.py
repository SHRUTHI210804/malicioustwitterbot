import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(page_title="Detecting Malicious Twitter Bots", layout="wide")

# UI STYLE
st.markdown("""
<style>

/* ===== BACKGROUND DIM EFFECT ===== */
.stApp{
background-size:cover;
background-attachment:fixed;
}

/* ===== MAIN HEADING ===== */
.main-heading{
background:black;
padding:20px;
border-radius:12px;
text-align:center;
color:white;
font-size:40px;
font-weight:bold;
box-shadow:0px 0px 30px rgba(255,255,255,0.9);
text-shadow:0px 0px 15px rgba(255,255,255,1);
}

.sub-heading{
text-align:center;
color:white;
font-size:20px;
text-shadow:0px 0px 10px rgba(255,255,255,0.8);
}

/* ===== BUTTON STYLE ===== */
.stButton>button{
background:linear-gradient(90deg,#00c6ff,#0072ff);
color:white;
font-size:18px;
font-weight:bold;
padding:12px 28px;
border-radius:12px;
border:none;
box-shadow:0px 0px 18px rgba(0,114,255,0.8);
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.08);
background:linear-gradient(90deg,#ff512f,#dd2476);
}

/* ===== UPLOAD BOX ===== */
[data-testid="stFileUploader"] section{
background:rgba(0,0,0,0.7);
border-radius:12px;
border:3px dashed #00ffff;
padding:20px;
}

/* ===== CONTAINER ===== */
.block-container{
background-color:rgba(0,0,0,0.8);
padding:2rem;
border-radius:15px;
box-shadow:0px 0px 20px rgba(255,255,255,0.3);
}

/* ===== TEXT HIGHLIGHT ===== */
h1,h2,h3,h4,h5,p,label,div{
color:white!important;
text-shadow:0px 0px 8px rgba(255,255,255,0.8);
}

</style>
""", unsafe_allow_html=True)

# BACKGROUND IMAGE WITH DULL EFFECT
def set_background(image):
    try:
        with open(image, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(0,0,0,0.75),
                rgba(0,0,0,0.75)
            ),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)

    except:
        st.warning("Background image not found")

set_background("IMAGE.jpg")

# TITLE
st.markdown("""
<div class="main-heading">
🛡 Detecting Malicious Twitter Bots
</div>
<div class="sub-heading">
Behavioral Analysis Based Social Media Bot Detection
</div>
""", unsafe_allow_html=True)

# FILE UPLOAD
uploaded_file = st.file_uploader("📂 Upload Twitter Dataset", type=["csv"])

if uploaded_file:

    try:
        dataset = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        dataset = pd.read_csv(uploaded_file, encoding="latin1")

    st.success("✅ Dataset Loaded Successfully")
    st.dataframe(dataset.head())

    required_columns = [
        'followers_count','friends_count','listedcount',
        'favourites_count','statuses_count',
        'verified','bot','screen_name'
    ]

    if all(col in dataset.columns for col in required_columns):

        X = dataset[['followers_count','friends_count','listedcount',
                     'favourites_count','statuses_count','verified']].copy()

        X['verified'] = X['verified'].astype(int)
        y = dataset['bot']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # MODEL ANALYSIS
        if st.button("📊 Run Model Performance Analysis"):

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
            }

            for name, model in models.items():

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, pred)
                precision = precision_score(y_test, pred)
                recall = recall_score(y_test, pred)
                f1 = f1_score(y_test, pred)
                auc = roc_auc_score(y_test, pred)

                st.subheader(name)

                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{accuracy*100:.2f}%")
                c2.metric("Precision", f"{precision*100:.2f}%")
                c3.metric("Recall", f"{recall*100:.2f}%")

                st.write("F1:", f1)
                st.write("AUC:", auc)

                fpr, tpr, _ = roc_curve(y_test, pred)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr)
                ax.plot([0,1],[0,1],'r--')
                ax.set_title(name + " ROC")
                st.pyplot(fig)

        # BEST MODEL SELECTION
        best_model = None
        best_score = 0
        best_name = ""

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            f1 = f1_score(y_test, pred)

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

        final_model = best_model
        st.success(f"✅ Best Model Selected: {best_name}")

        # USER SCAN
        st.markdown("---")
        st.header("🔍 Social Media Intelligence Deep Scanner")

        username_input = st.text_input("Enter Twitter Username")

        if st.button("🚀 Run Deep Scan"):

            dataset['screen_name_lower'] = dataset['screen_name'].astype(str).str.lower()

            if username_input.lower() in dataset['screen_name_lower'].values:

                user_data = dataset[
                    dataset['screen_name_lower'] == username_input.lower()
                ].iloc[0]

                st.markdown("### 👤 User Account Details")

                user_df = pd.DataFrame(user_data).reset_index()
                user_df.columns = ["Feature", "Value"]
                st.dataframe(user_df, use_container_width=True)

                st.markdown("### 📊 Behavioral Intelligence Card")

                c1, c2, c3 = st.columns(3)
                c1.metric("Followers", user_data['followers_count'])
                c2.metric("Friends", user_data['friends_count'])
                c3.metric("Statuses", user_data['statuses_count'])

                ratio = user_data['followers_count']/(user_data['friends_count']+1)
                st.metric("Follower-Friend Ratio", f"{ratio:.2f}")

                input_features = np.array([[ 
                    user_data['followers_count'],
                    user_data['friends_count'],
                    user_data['listedcount'],
                    user_data['favourites_count'],
                    user_data['statuses_count'],
                    int(user_data['verified'])
                ]])

                prediction = final_model.predict(input_features)
                probability = final_model.predict_proba(input_features)

                bot_prob = probability[0][1]*100
                real_prob = probability[0][0]*100

                st.markdown("### 🧠 AI Intelligence Report")

                c1, c2 = st.columns(2)
                c1.metric("Authenticity", f"{real_prob:.2f}%")
                c2.metric("Threat Probability", f"{bot_prob:.2f}%")

                st.progress(int(bot_prob))

                if prediction[0] == 1:
                    st.error("🚨 BOT ACCOUNT DETECTED")
                else:
                    st.success("✅ AUTHENTIC HUMAN ACCOUNT")

            else:
                st.warning("Username not found")

    else:
        st.error("Dataset missing required columns")
