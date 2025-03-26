# Spam Classifier using Naïve Bayes

## 📌 Project Overview

This project is a **spam classifier** that detects whether a given message is **spam** or **ham** (not spam). It uses **Naïve Bayes** (MultinomialNB) and Natural Language Processing (NLP) techniques to classify text messages.

## 🚀 Features

- Preprocesses text data (removes stopwords, punctuation, etc.)
- Uses **CountVectorizer** for feature extraction
- Trains a **Multinomial Naïve Bayes** model
- Predicts whether a message is **spam or ham**
- Streamlit-based web interface for user input

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 📂 Dataset

The dataset used for training consists of labeled text messages:

- `label` (0 for ham, 1 for spam)
- `text` (actual message content)

## 📜 Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/your-repo/spam-classifier.git
   cd spam-classifier
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```sh
   streamlit run app.py
   ```

## 🏗 Usage

1. Enter a text message into the input field.
2. Click **Predict**.
3. The model will classify the message as **Spam** or **Ham**.

## 🎯 Model Training

To retrain the model:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("spam.csv")

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
```

## 🛡 License

This project is open-source and free to use.

---

**Contributors:** Mohammed zahid

Feel free to modify this README as needed! 🚀

