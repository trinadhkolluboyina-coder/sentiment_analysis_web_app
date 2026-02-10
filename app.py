# ================================
# Core Packages
# ================================
import streamlit as st
import altair as alt
import plotly.express as px

# ================================
# EDA Packages
# ================================
import pandas as pd
import numpy as np
from datetime import datetime

# ================================
# Utils
# ================================
import joblib

# Load Model
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr_10_02_2026.pkl", "rb"))

# ================================
# Tracking Utils
# ================================
from track_utils import (
    create_page_visited_table,
    add_page_visited_details,
    view_all_page_visited_details,
    add_prediction_details,
    view_all_prediction_details,
    create_emotionclf_table
)

# ================================
# Prediction Functions
# ================================
def predict_emotions(text):
    return pipe_lr.predict([text])[0]

def get_prediction_proba(text):
    return pipe_lr.predict_proba([text])

# ================================
# Emoji Mapping
# ================================
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# ================================
# Main App
# ================================
def main():
    st.set_page_config(
        page_title="Emotion Classifier",
        page_icon="😊",
        layout="wide"
    )

    st.title("🧠 Emotion Classifier App")

    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize DB tables
    create_page_visited_table()
    create_emotionclf_table()

    # ================================
    # HOME
    # ================================
    if choice == "Home":
        add_page_visited_details("Home", datetime.now())
        st.subheader("🔍 Emotion Analysis of Text")

        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area("Enter your text here")
            submit_text = st.form_submit_button(label="Analyze Emotion")

        if submit_text:
            if raw_text.strip() == "":
                st.warning("⚠️ Please enter some text to analyze.")
                return

            col1, col2 = st.columns(2)

            # Predictions
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            confidence = np.max(probability)

            # Save prediction
            add_prediction_details(
                raw_text,
                prediction,
                confidence,
                datetime.now()
            )

            # ================================
            # Left Column
            # ================================
            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.markdown(f"### {prediction.capitalize()} {emoji_icon}")
                st.write(f"**Confidence:** {confidence:.2%}")

            # ================================
            # Right Column
            # ================================
            with col2:
                st.success("Prediction Probability")

                proba_df = pd.DataFrame(
                    probability,
                    columns=pipe_lr.classes_
                ).T.reset_index()

                proba_df.columns = ["Emotion", "Probability"]

                chart = (
                    alt.Chart(proba_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Emotion", sort="-y"),
                        y="Probability",
                        color="Emotion"
                    )
                )

                st.altair_chart(chart, use_container_width=True)

    # ================================
    # MONITOR
    # ================================
    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.subheader("📊 App Monitoring Dashboard")

        with st.expander("📌 Page Metrics"):
            page_visited_details = pd.DataFrame(
                view_all_page_visited_details(),
                columns=["Page", "Time"]
            )

            st.dataframe(page_visited_details)

            pg_count = (
                page_visited_details["Page"]
                .value_counts()
                .reset_index()
            )
            pg_count.columns = ["Page", "Visits"]

            bar_chart = alt.Chart(pg_count).mark_bar().encode(
                x="Page",
                y="Visits",
                color="Page"
            )
            st.altair_chart(bar_chart, use_container_width=True)

            pie_chart = px.pie(
                pg_count,
                values="Visits",
                names="Page"
            )
            st.plotly_chart(pie_chart, use_container_width=True)

        with st.expander("😊 Emotion Classifier Metrics"):
            df_emotions = pd.DataFrame(
                view_all_prediction_details(),
                columns=["Text", "Prediction", "Confidence", "Time"]
            )

            st.dataframe(df_emotions)

            pred_count = (
                df_emotions["Prediction"]
                .value_counts()
                .reset_index()
            )
            pred_count.columns = ["Emotion", "Count"]

            emotion_bar = alt.Chart(pred_count).mark_bar().encode(
                x="Emotion",
                y="Count",
                color="Emotion"
            )
            st.altair_chart(emotion_bar, use_container_width=True)

    # ================================
    # ABOUT
    # ================================
    else:
        add_page_visited_details("About", datetime.now())
        st.subheader("ℹ️ About This App")
        st.write(
            """
            This Emotion Classifier App analyzes user text and predicts
            emotional states using a machine learning model trained on
            labeled emotional data.

            **Tech Stack:**
            - Streamlit
            - Scikit-learn
            - Altair & Plotly
            - Joblib
            """
        )


# ================================
# Run App
# ================================
if __name__ == "__main__":
    main()
