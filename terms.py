from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer

# Download stopwords from nltk
nltk.download('stopwords')

app = Flask(__name__)

# Set of English stopwords
STOP_WORDS = set(stopwords.words('english'))

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L12-v2')
d = model.get_sentence_embedding_dimension()
assert d == 384

# HTML Template to render the form and common terms tables
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>CSV Upload and Common Terms</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #007BFF;
            color: white;
        }
        td {
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <h1>CSV File Data with Common Terms</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="csvfile">Upload CSV file:</label>
        <input type="file" name="csvfile" id="csvfile" accept=".csv">
        <br><br>
        <button type="submit">Upload and View Common Terms</button>
    </form>

    {% for category, terms in terms_data %}
    <h2>Top 5 Common Terms for {{ category }}</h2>
    <table border="1">
        <thead>
            <tr>
                <th>Phrase</th>
                <th>Frequency</th>
            </tr>
        </thead>
        <tbody>
        {% for term, count in terms %}
            <tr>
                <td>{{ term }}</td>
                <td>{{ count }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% endfor %}

    <h2>Pairwise Cosine Similarity Matrix</h2>
    <table border="1">
        {% for row in similarity_matrix %}
        <tr>
            {% for value in row %}
            <td>{{ "%.2f" | format(value) }}</td>
            {% endfor %}
        </tr>
        {% endfor %}
    </table>
    {% endif %}
</body>
</html>
"""

def extract_common_terms(df, column='summary', n=8):
    """Extract the top n most common bigrams or trigrams from the summary column."""
    if column in df.columns:
        # Join all the summary text
        all_summaries = " ".join(df[column].dropna())

        # Clean the text: remove punctuation and split into words
        words = re.findall(r'\b\w+\b', all_summaries.lower())

        # Remove stop words
        filtered_words = [word for word in words if word not in STOP_WORDS]

        # Generate 2-grams (bigrams) or 3-grams (trigrams)
        bigrams = ngrams(filtered_words, 2)  # Use 3 for trigrams
        bigram_counts = Counter(bigrams)

        # Get the top n most common phrases (as bigrams or trigrams)
        common_terms = bigram_counts.most_common(n)

        # Convert tuples of n-grams back to strings for display
        return [(" ".join(term), count) for term, count in common_terms]
    return []

@app.route("/", methods=["GET", "POST"])
def index():
    terms_data = []
    if request.method == "POST":
        file = request.files.get("csvfile")
        
        if file and file.filename.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)

            # Generate common terms for ALL, MALE, and FEMALE categories
            for category, sex_value in [("ALL", None), ("MALE", "MALE"), ("FEMALE", "FEMALE")]:
                if sex_value:
                    filtered_df = df[df['sex'] == sex_value]
                else:
                    filtered_df = df  # ALL

                # Extract the common terms from the 'summary' column
                common_terms = extract_common_terms(filtered_df)

                # Add the common terms for this category
                terms_data.append((category, common_terms))

             # Calculate embeddings
            n = len(terms_data)
            embed = np.array(model.encode(terms_data))

            # Pairwise cosine similarity matrix
            similarity_matrix = np.array([[1.0 - cosine(embed[i], embed[j]) for j in range(n)] for i in range(n)])

            # Prepare data for display
            topic_embeddings = [(topic, embed[i][:5].tolist()) for i, topic in enumerate(terms_data)]
    
    # Render the HTML page, passing in the common terms for ALL, MALE, and FEMALE
    return render_template_string(html_template, terms_data=terms_data, topic_embeddings=topic_embeddings, similarity_matrix=similarity_matrix)

if __name__ == "__main__":
    app.run(debug=True)