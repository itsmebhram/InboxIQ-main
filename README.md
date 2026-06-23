# SMS Spam Detector 🚨

An AI-powered SMS spam detection web app built with Flask, Scikit-learn, and NLP.

## Features
- 98%+ accuracy using Naive Bayes + TF-IDF
- Confidence score with visual progress bar
- Suspicious keyword detection (Explainable AI)
- Message analysis - word count, caps ratio, special characters
- Clean dark-themed UI

## Tech Stack
- **Backend:** Python, Flask
- **ML:** Scikit-learn (Naive Bayes, TF-IDF Vectorizer)
- **Dataset:** SMS Spam Collection (5,572 messages)
- **Frontend:** HTML, CSS, JavaScript

## Run Locally

```bash
git clone https://github.com/yourusername/spam-detector
cd spam-detector
pip install -r requirements.txt
python train_model.py
python app.py
```

Visit `http://localhost:5000`

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 98.21% |
| Precision (Spam) | 99% |
| Recall (Spam) | 87% |

## How It Works
1. Message is cleaned (lowercased, special chars removed)
2. TF-IDF vectorizer converts text to numerical features
3. Naive Bayes classifier predicts spam/ham
4. Confidence score shown via `predict_proba()`
5. Top spam-triggering keywords highlighted
