# Movie-Review-Sentiment-Analyzer

Movie Review Sentiment Analyzer
A full-stack web application built with Python and Flask that classifies movie reviews as positive or negative using Natural Language Processing (NLP) techniques.
![Sentiment Analysis Project Banner/Screenshot - Placeholder for an image of your running application interface] (You can add a screenshot here later for better presentation)

🚀 Technologies Used
This project leverages the following professional technology stack:
Backend Framework: Flask (Python)
Data Manipulation: Pandas
NLP & ML Libraries: NLTK, Scikit-learn
Vectorization: TF-IDF (TfidfVectorizer)
Model: Logistic Regression
Frontend: HTML5, CSS3 (Static files)

✨ Features
User-Friendly Interface: Simple web form to submit a movie review.
Real-time Analysis: Instant classification of text sentiment (Positive/Negative).
Robust Preprocessing: Includes stop-word removal, stemming, and HTML tag cleaning.
Professional Structure: Organized into modular Python files for maintainability (app.py, model.py, preprocessing.py).

📊 Model Performance
The trained Logistic Regression model achieved an accuracy of X% on the test dataset.
(Note: Update X% with the actual accuracy score you obtained when running model.py.)
🛠️ Installation and Setup
Follow these steps to get the project up and running on your local machine.
Prerequisites
You need Python 3.8+ installed.
Step 1: Clone the Repository
Clone this GitHub repository to your local machine using the terminal:
bash
git clone github.com
cd movie-review-sentiment-analyzer
Use code with caution.

Step 2: Set up Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to manage project dependencies:
bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment:
# On Windows (Command Prompt):
.\venv\Scripts\activate

# On macOS and Linux:
source venv/bin/activate
Use code with caution.

Step 3: Install Dependencies
Install all required Python libraries listed in requirements.txt:
bash
pip install -r requirements.txt
Use code with caution.

Step 4: Train the ML Model
Before running the web application, you must train the machine learning model and save the required .pkl files (sentiment_model.pkl and tfidf_vectorizer.pkl). Run the following command once:
bash
python model.py
Use code with caution.

This script will also download necessary NLTK data (stopwords, punkt).
🏃 Usage
Running the Application
Start the Flask development server:
bash
python app.py
Use code with caution.

The application will now be running. Open your web browser and navigate to:
http://127.0.0.1:5000/
Using the Interface
Enter a movie review into the text area on the homepage.
Click the Analyze Sentiment button.
The results page will display the predicted sentiment (Positive or Negative).
📁 Project Structure
/sentiment_analyzer
|-- app.py                # Main Flask application logic
|-- model.py              # Model training script
|-- preprocessing.py      # Text cleaning utilities
|-- requirements.txt      # Project dependencies
|-- README.md             # This file
|-- *.pkl                 # Generated model files (after running model.py)
|-- /static               
|   |-- style.css         # Basic styling
|-- /templates            
|   |-- index.html        # Input form page
|   |-- results.html      # Sentiment results page
🤝 Contributing
If you have suggestions or want to improve the model accuracy, feel free to fork the repository and submit a pull request.
Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is licensed under the MIT License.
📞 Contact
Name - kamesh p
Email Address - kamesh1725@gmail.com
Project Link:
