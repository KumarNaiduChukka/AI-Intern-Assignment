import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load model and vectorizer
try:
    model = joblib.load('P:\\Soulpage IT Solutions\\ASS\\Models\\sentiment_model.pkl')
    tfidf = joblib.load('P:\\Soulpage IT Solutions\\ASS\\Models\\tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are in the correct directory.")
    st.stop()

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title('Flipkart Review Sentiment Analysis')

review = st.text_area("Enter your review:")

if st.button('Analyze Sentiment'):
    # Preprocess
    cleaned_review = preprocess_text(review)
    # Vectorize
    review_tfidf = tfidf.transform([cleaned_review])
    # Predict
    prediction = model.predict(review_tfidf)[0]
    # Display result
    st.subheader('Prediction Result')
    if prediction == 'positive':
        st.success('Positive Sentiment 😊')
    elif prediction == 'neutral':
        st.info('Neutral Sentiment 😐')
    else:
        st.error('Negative Sentiment 😠')