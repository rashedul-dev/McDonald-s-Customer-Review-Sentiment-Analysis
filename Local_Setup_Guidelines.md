# 🍟 McDonald's Customer Review Sentiment Analysis  

This project performs **sentiment analysis** on McDonald's customer reviews using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.  
It classifies reviews into three categories: **Positive**, **Neutral**, and **Negative**.

---

## 📁 Project Structure

```
mcdonalds-sentiment-analysis/
│
├── 📁 data/
│   └── mcd_reviews.csv               # Dataset file
│
├── 📁 notebooks/
│   └── sentiment_analysis.ipynb      # Jupyter notebook with code and experiments
│
├── 📁 models/
│   ├── sentiment_model.joblib        # Saved best ML model (LinearSVC)
│   └── vectorizer.joblib             # Saved TF-IDF vectorizer
│
├── 📁 results/
│   └── evaluation_report.txt         # Accuracy and performance metrics
│
└── README.md                         # Project documentation
```

---

## ⚙️ Local Setup Instructions

Follow these steps to set up the project locally and run sentiment analysis on your machine.

---

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/mcdonalds-sentiment-analysis.git
cd mcdonalds-sentiment-analysis
```

---

### 2️⃣ Install Python and dependencies

Make sure you have **Python 3.8+** installed.  
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

> **Note:** If you don’t have a `requirements.txt` file yet, you can install the libraries manually:
```bash
pip install pandas numpy matplotlib scikit-learn joblib nltk
```

---

### 3️⃣ Download NLTK Data

Open Python and run:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

> This downloads the necessary NLP resources for text preprocessing.

---

### 4️⃣ Prepare Dataset

Place your McDonald’s reviews CSV file inside the `data/` folder:

```
data/mcd_reviews.csv
```

The CSV should contain at least the following columns:  
`review`, `rating`, and `label` (POSITIVE, NEUTRAL, NEGATIVE).

---

### 5️⃣ Run the Jupyter Notebook (Optional)

If you want to explore the code step by step:

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

This notebook includes:
- Data loading
- Cleaning and preprocessing
- Model training and evaluation
- Sample predictions

---

### 6️⃣ Run Sentiment Analysis Script

If you want to run predictions directly using the pre-trained model:

```python
import joblib
from sentiment_analysis import predict_mcdonalds_sentiment

# Load model and vectorizer
model = joblib.load('models/sentiment_model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

# Test a review
review = "The burger was delicious and the service was fast!"
result = predict_mcdonalds_sentiment(review, model=model, vectorizer=vectorizer)
print(result)
```

Expected output:

```txt
{
    'sentiment': 'POSITIVE',
    'confidence': 0.792,
    'cleaned_text': 'burger delicious service fast'
}
```

---

### 7️⃣ Batch Prediction

You can analyze multiple reviews at once:

```python
reviews = [
    "Fast service and tasty food",
    "Rude staff and long wait times",
    "The ice cream machine was working for once!"
]

from sentiment_analysis import analyze_multiple_reviews
batch_results = analyze_multiple_reviews(reviews)
print(batch_results)
```

---

### 8️⃣ Evaluation Metrics

Check the model performance in `results/evaluation_report.txt`:
- Accuracy  
- Sentiment distribution  
- Model comparison  

---

### 9️⃣ Optional Improvements

- Train on larger or multilingual datasets  
- Integrate with a web app (Flask/Streamlit)  
- Visualize trends with matplotlib/seaborn

---

💡 **Congratulations!** Your sentiment analysis environment is now set up locally.  
You can now test new reviews, fine-tune the model, or explore the dataset further.

