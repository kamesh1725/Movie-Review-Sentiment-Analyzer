import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- NLTK Data Management ---
# Define a path within your project directory for NLTK data (professional practice)
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data to the specified path
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
except Exception as e:
    print(f"NLTK download failed: {e}")

# Initialize NLTK components globally
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and prepares text for analysis using a simple regex tokenizer."""
    # Ensure input is string
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r'<[^>]+>', '', text)          # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # Keep only letters and spaces
    text = text.lower()

    # Use simple splitting by space instead of complex nltk.word_tokenize
    tokens = text.split() 

    # Remove stopwords and apply stemming
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def get_tfidf_vectorizer():
    """Returns an initialized TF-IDF Vectorizer instance configured for N-grams."""
    # MODIFIED: Added ngram_range=(1, 2) to capture bigrams like "very good"
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
