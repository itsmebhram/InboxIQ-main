from flask import Flask, render_template, request, jsonify
import pickle
import re
import numpy as np

app = Flask(__name__)

SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'won', 'prize', 'claim', 'urgent', 'click',
    'call now', 'limited', 'offer', 'cash', 'reward', 'selected', 'congrats',
    'congratulations', 'ringtone', 'txt', 'mobile', 'stop', 'reply', 'guaranteed'
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_suspicious_keywords(text):
    text_lower = text.lower()
    found = []
    for kw in SPAM_KEYWORDS:
        if kw in text_lower and kw.upper() not in found:
            found.append(kw.upper())
    return found[:6]

def analyze_message(text):
    words = text.split()
    chars = len(text)
    special_chars = len(re.findall(r'[!?$#@%&*]', text))
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = round((caps_count / chars * 100) if chars > 0 else 0, 1)
    return {
        'word_count': len(words),
        'char_count': chars,
        'special_chars': special_chars,
        'caps_ratio': caps_ratio
    }

@app.route('/ping', methods=['GET', 'HEAD'])
def ping():
    return "OK", 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    # Load models on demand to save initial RAM
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = round(float(np.max(proba)) * 100, 1)

    return jsonify({
        'is_spam': bool(prediction),
        'confidence': confidence,
        'label': 'SPAM' if prediction else 'NOT SPAM',
        'keywords': get_suspicious_keywords(message) if prediction else [],
        'stats': analyze_message(message)
    })

if __name__ == '__main__':
    app.run(debug=True)
