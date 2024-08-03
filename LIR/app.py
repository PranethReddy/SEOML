import requests
from bs4 import BeautifulSoup

def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('title').get_text() if soup.find('title') else ''
    meta_desc = soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else ''
    body_content = ' '.join([p.get_text() for p in soup.find_all('p')])
    return {'title': title, 'meta_description': meta_desc, 'body_content': body_content}
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_and_extract_features(data):
    documents = [data['title'], data['meta_description'], data['body_content']]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix
from sklearn.linear_model import LinearRegression
import joblib

# Example data
data = {
    'Keyword': ['seo tips', 'keyword research', 'content marketing'],
    'Search Volume': [5000, 3000, 4000]
}

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Keyword'])
y = df['Search Volume']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'model/keyword_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/keyword_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form['url']
    content = scrape_content(url)
    _, tfidf_matrix = preprocess_and_extract_features(content)
    keywords = vectorizer.inverse_transform(tfidf_matrix)
    keyword_suggestions = {keyword: int(model.predict(vectorizer.transform([keyword]))[0]) for keyword in keywords[0]}
    return render_template('result.html', keywords=keyword_suggestions)

if __name__ == '__main__':
    app.run(debug=True)
