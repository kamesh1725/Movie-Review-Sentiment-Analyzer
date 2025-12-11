import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
# Import the preprocessing functions we created in Step 2/Improvement steps
from preprocessing import preprocess_text, get_tfidf_vectorizer

def train_model():
    print("Starting model training process...")

    # --- 1. Load Data ---
    # Make sure 'IMDB_Dataset.csv' is in your project directory
    # If you are using a specific absolute path, replace 'IMDB_Dataset.csv' with your path:
    # Example: df = pd.read_csv("C:/Users/kamesh.p/OneDrive/Desktop/MY_PROJECTS/Sentiment Analysis/IMDB Dataset.csv")
    try:
        df = pd.read_csv('C:/Users/kamesh.p/OneDrive/Desktop/MY_PROJECTS/Sentiment Analysis/IMDB Dataset.csv')
    except FileNotFoundError:
        print("Error: IMDB_Dataset.csv not found in the current directory.")
        print("Please check your file path or download the dataset.")
        return

    # Use a subset of data for development speed (20% of 50k reviews = 10k reviews)
    df = df.sample(frac=0.2, random_state=42) 

    # Convert 'sentiment' column to binary (1 for positive, 0 for negative)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    X = df['review']
    y = df['sentiment']

    # --- 2. Preprocess Data ---
    print("Preprocessing text data...")
    # Apply the cleaning function from preprocessing.py
    X_cleaned = X.apply(preprocess_text)

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)

    # --- 4. Vectorize Data (TF-IDF) ---
    # This step now uses n-grams (1 and 2 words) due to your update in preprocessing.py
    print("Vectorizing text data using TF-IDF N-grams...")
    tfidf_vectorizer = get_tfidf_vectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # --- 5. Train Model ---
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # --- 6. Evaluate Model ---
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

    # --- 7. Save Model and Vectorizer (using joblib) ---
    model_path = 'sentiment_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")

# This ensures the function runs automatically when you execute the file
if __name__ == "__main__":
    train_model()
