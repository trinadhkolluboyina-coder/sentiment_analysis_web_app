# 🧠 Emotion Analysis Web App (NLP)

This project is an **Emotion Analysis Web Application** built using **Natural Language Processing (NLP)** and **Machine Learning**, deployed using **Streamlit**.  
It classifies user-input text into emotional categories such as *joy, sadness, anger, fear, surprise,* and *neutral*.

---

## 🚀 Features

- 🔍 Emotion classification from raw text
- 📊 Probability distribution visualization
- 😊 Emoji-based emotion representation
- 📈 App usage & emotion monitoring dashboard
- 🗃️ Persistent tracking using SQLite
- 🧪 Trained Logistic Regression pipeline

---

## 🗂️ Project Structure

Emotion-analysis_NLP/
│
├── app.py # Streamlit web app
├── track_utils.py # SQLite tracking utilities
├── emotion_classifier_pipe_lr_10_02_2026.pkl # Trained ML model
├── emotion analysis.ipynb # Model training & EDA
├── emotion_dataset_2.csv # Emotion-labeled dataset
├── requirements.txt
└── README.md



📊 Monitoring & Analytics

The app tracks:

Page visits (Home / Monitor / About)

User emotion predictions

Confidence scores

Timestamps

All data is stored locally using SQLite (emotion_data.db).

🧪 Dataset

File: emotion_dataset_2.csv

Format: Text–Emotion labeled data

Used for training and evaluation in emotion analysis.ipynb

🔮 Future Improvements

Transformer-based models (BERT, RoBERTa)

Multi-label emotion detection

Sentence-level emotion shifts

REST API using FastAPI

Docker deployment

Multilingual emotion analysis

📜 License

This project is for educational and research purposes.
Feel free to use, modify, and extend it.

🙌 Acknowledgements

Streamlit

scikit-learn

Altair & Plotly

Open-source NLP community
