import os
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, Markup

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Function for preprocessing documents
def preprocess(text):
    text = text.lower()  # Case folding
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize text
    text = re.sub(r'[^\w\s]', '', text)  # Normalize text
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to load documents and their titles
def load_documents(folder_path):
    documents = []
    document_titles = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                document_title = os.path.splitext(filename)[0]
                document_content = preprocess(file.read())  # Mengambil konten dokumen
                documents.append(document_content)
                document_titles.append(document_title)
    return documents, document_titles


# Load and preprocess documents along with their titles
folder_path = 'documents'
documents, document_titles = load_documents(folder_path)

# Create TF-IDF model and transform documents into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Create index and posting list
terms = vectorizer.get_feature_names_out()
index = {term: X[:, i].nonzero()[0].tolist() for i, term in enumerate(terms)}

# Function to process query
def process_query(query):
    query = preprocess(query)
    query_vector = vectorizer.transform([query])
    return query_vector

# Function to calculate similarity and rank documents
def rank_documents(query_vector, X):
    similarities = cosine_similarity(query_vector, X)
    ranked_indices = similarities.argsort()[0][::-1]
    return ranked_indices, similarities

# Function to display top documents
def display_top_documents(ranked_indices, similarities, top_n):
    top_docs = [{'index': doc_index, 'score': similarities[0, doc_index], 'content': documents[doc_index][:200]}
                for doc_index in ranked_indices[:top_n]]
    return top_docs

# Function to highlight query in text
def highlight_query(text, query):
    highlighted_text = text.replace(query, f'<span class="highlight">{query}</span>')
    return Markup(highlighted_text)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        query_vector = process_query(query)
        ranked_indices, similarities = rank_documents(query_vector, X)
        top_docs_3 = display_top_documents(ranked_indices, similarities, 3)
        top_docs_5 = display_top_documents(ranked_indices, similarities, 5)
        top_docs_10 = display_top_documents(ranked_indices, similarities, 10)
        return render_template('index.html', query=query, top_docs_3=top_docs_3, top_docs_5=top_docs_5, top_docs_10=top_docs_10, highlight_query=highlight_query, document_titles=document_titles)
    return render_template('index.html', query='', top_docs_3=[], top_docs_5=[], top_docs_10=[], document_titles=document_titles)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
