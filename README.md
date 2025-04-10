## Customer Analytics Dashboard

This project is a Streamlit-based web application that provides interactive tools for:

Customer Segmentation
Churn Prediction
Sentiment Analysis on Tweets

## 🚀 Features

# 1. Customer Segmentation

Visualize customer data (e.g., MonthsInService).
Use KMeans clustering to segment customers.
Visualize the optimal number of clusters using the Elbow Method.

# 2. Churn Analysis

Cleans and preprocesses customer data.
Encodes categorical features and imputes missing values.
Trains a machine learning model (XGBoost) to predict customer churn.
Evaluates model performance using classification_report.

# 3. Sentiment Analysis

Loads and preprocesses a tweet dataset.
Removes stopwords and punctuation.
Applies lemmatization with spaCy.
Vectorizes text using TF-IDF.
Trains a Naive Bayes classifier to detect sentiment.

## 🛠️ Installation

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

Make sure to place the required datasets in the project root:
cell2cellholdout.csv
tweetsdataset.csv

## 📊 How to Run

streamlit run app.py
Then, go to http://localhost:8501/ in your browser.

## 📁 File Structure

├── app.py
├── cell2cellholdout.csv
├── tweetsdataset.csv
├── README.md
└── requirements.txt

## 📦 Requirements

streamlit
pandas, numpy
scikit-learn
xgboost
plotly, seaborn, matplotlib
nltk, spacy
Run python -m nltk.downloader stopwords and python -m spacy download en_core_web_sm to download required language data.