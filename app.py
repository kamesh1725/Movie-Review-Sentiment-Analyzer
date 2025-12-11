from flask import Flask, render_template, request
import joblib
import os
from preprocessing import preprocess_text 

# --- Configuration ---
app = Flask(__name__)
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# Define neutral sentiment thresholds
NEUTRAL_THRESHOLD_LOW = 0.4
NEUTRAL_THRESHOLD_HIGH = 0.6

# --- Load Model and Vectorizer ---
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model files not found.")
    exit(1)

# --- Flask Routes (Endpoints) ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review_text']

        # 1. Preprocess and Vectorize
        cleaned_text = preprocess_text(review_text)
        text_vectorized = vectorizer.transform([cleaned_text])

        # 2. Get the prediction PROBABILITY (the confidence score)
        # model.predict_proba returns [[prob_negative, prob_positive]]
        prediction_proba = model.predict_proba(text_vectorized)[0][1] # Get just the positive probability

        # 3. Classify based on thresholds
        if prediction_proba >= NEUTRAL_THRESHOLD_HIGH:
            sentiment_label = 'Positive 😊'
            sentiment_class = 'Positive'
        elif prediction_proba <= NEUTRAL_THRESHOLD_LOW:
            sentiment_label = 'Negative 😠'
            sentiment_class = 'Negative'
        else:
            sentiment_label = 'Neutral 😐' # The new 'Average' option
            sentiment_class = 'Neutral'
        
        # Pass data to the results template
        return render_template('results.html', 
                               original_review=review_text, 
                               sentiment_label=sentiment_label,
                               sentiment_class=sentiment_class,
                               confidence=f"{prediction_proba*100:.2f}%") # Display confidence score

if __name__ == '__main__':
    app.run(debug=True)
