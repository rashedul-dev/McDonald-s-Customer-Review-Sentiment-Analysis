# 🍟 McDonald's Customer Review Sentiment Analysis

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze customer reviews of **McDonald's** and classify them into three sentiment categories:

- 🟢 **Positive**
- 🟡 **Neutral** 
- 🔴 **Negative**

By processing over **33,000 real-world reviews**, this project demonstrates how text data can be transformed into actionable insights for businesses to improve their services and customer satisfaction.

---

## 📌 Project Overview

The goal of this project is to build a machine learning model that accurately predicts the sentiment of McDonald's customer reviews. It covers the full pipeline — from **data preprocessing and feature extraction** to **model training, evaluation, and deployment-ready prediction**.

### 🔎 Key Outcomes

- Automatic classification of user reviews with **~82% accuracy**
- Detailed insights into customer satisfaction trends  
- Production-ready sentiment prediction model

---

## 📂 Dataset Information

- **Source:** Local CSV file containing scraped McDonald's customer reviews
- **Total Reviews:** 33,396
- **Columns:**
  - `reviewer_id`, `store_name`, `category`, `store_address`, `latitude`, `longitude`, `rating_count`, `review_time`, `review`, `rating`, `label`
- **Target Labels Distribution:**
  - 🟢 `POSITIVE` – 16,061
  - 🔴 `NEGATIVE` – 12,517
  - 🟡 `NEUTRAL` – 4,818
s
---

## 🧹 Data Preprocessing Steps

The dataset undergoes the following NLP preprocessing steps before model training:

1. ✅ Convert text to lowercase
2. 🧼 Remove HTML tags, URLs, special characters
3. ✂️ Tokenization (splitting text into words)
4. 📚 Lemmatization (word normalization)
5. 🚫 Stopword removal (removing common words like *the, is, and*)
6. 📊 TF-IDF vectorization (converting text into numerical features)

---

## 🧠 Machine Learning Models

Three classic ML models were trained and evaluated for sentiment classification:

| Model | Accuracy |
|-------|----------|
| **LinearSVC** | 🏆 **0.8195** |
| Logistic Regression | 0.8162 |
| Multinomial Naive Bayes | 0.8069 |

✅ **Best Model:** `LinearSVC` — delivering the highest accuracy with strong performance across all sentiment classes.

---

## 📊 Model Performance Summary

| Metric | Value |
|--------|-------|
| **Total Reviews** | 33,396 |
| **Positive** | 16,061 |
| **Neutral** | 4,818 |
| **Negative** | 12,517 |
| **Best Model** | LinearSVC |
| **Accuracy** | 81.95% |

---
## 🔗 Quick Links

### 🤖 Setup Locally - GuideLines
- [View GuideLines](./Local_Setup_Guidelines.md)

---
## 📁 Project Structure
```
mcdonalds-sentiment-analysis/
│
├── 📁 data/
│ └── mcd_reviews.csv               # Dataset file
│
├── 📁 notebooks/
│ └── sentiment_analysis.ipynb      # Jupyter notebook with code and experiments
│
├── 📁 models/
│ ├── sentiment_model.joblib        # Saved best ML model (LinearSVC)
│ └── vectorizer.joblib             # Saved TF-IDF vectorizer
│
├── 📁 results/
│ └── evaluation_report.txt         # Accuracy and performance metrics
│
└── README.md                       # Project documentation

```
---

## 🚀 Future Improvements

- 📈 Train deep learning models (e.g., LSTM, BERT)
- 🌍 Deploy as a web app with Flask/Django or Streamlit
- 📊 Add data visualization dashboards for trends
- 🗣️ Add support for multilingual sentiment detection

---

## 📋 Models Summary Table

| Model Name | Library | Type | Usage |
|------------|---------|------|-------|
| LogisticRegression | sklearn.linear_model | Linear classifier | Baseline ML model for text |
| MultinomialNB | sklearn.naive_bayes | Probabilistic (Bayes) model | Very fast text classifier |
| LinearSVC | sklearn.svm | Support Vector Machine | Often most accurate |

---

## 💡 Conclusion

This project demonstrates how **NLP + Machine Learning** can extract meaningful insights from thousands of real-world customer reviews. With an accuracy of **~82%**, the sentiment analysis model provides a powerful foundation for automated feedback monitoring, business intelligence, and customer experience analytics.
