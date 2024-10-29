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
            background-color: #F8F9FA;
        }

        /* Define a color scale for background colors */
        .heatmap-cell {
            color: white;
            text-align: center;
        }
        /* Gradient colors for different ranges */
        .low { background-color: #f7fbff; color: black; }
        .medium-low { background-color: #c6dbef; }
        .medium { background-color: #6baed6; }
        .medium-high { background-color: #2171b5; }
        .high { background-color: #08306b; }
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
    {% if terms_data %}
    {% for category, terms in terms_data %}
    <h2>Top 8 Common Terms for {{ category }}</h2>
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
    {% endif %}
    {% if similarity_matrix %}
    <h2>Pairwise Cosine Similarity Matrix</h2>
    <table border="1">
        <th></th>
        {% for topic in topics %}
        <th>
        {{topic}}
        </th>
        {% endfor %}

        {% for row in similarity_matrix %}
        <tr>
            <th><strong>{{topics[loop.index0]}}</strong></th>
            {% for value in row %}
            <td class="heatmap-cell
                    {% if value < 0.15 %} low
                    {% elif value < 0.35 %} medium-low
                    {% elif value < 0.55 %} medium
                    {% elif value < 0.75 %} medium-high
                    {% else %} high
                    {% endif %}">{{ "%.2f" | format(value) }}</td>
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
        all_summaries = " ".join(df[column].dropna())
        words = re.findall(r'\b\w+\b', all_summaries.lower())
        filtered_words = [word for word in words if word not in STOP_WORDS]
        bigrams = ngrams(filtered_words, 2)  # Use 3 for trigrams if needed
        bigram_counts = Counter(bigrams)
        common_terms = bigram_counts.most_common(n)
        # Return both terms and counts directly
        return [(" ".join(term), count) for term, count in common_terms]
    return []

@app.route("/", methods=["GET", "POST"])
def index():
    terms_data = []
    similarity_matrix = None
    topics = []

    if request.method == "POST":
        file = request.files.get("csvfile")
        if file and file.filename.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)

            # Generate common terms for MALE and FEMALE categories only
            for category, sex_value in [("MALE", "MALE"), ("FEMALE", "FEMALE")]:
                filtered_df = df[df['sex'] == sex_value]
                common_terms = extract_common_terms(filtered_df)
                terms_data.append((category, common_terms))
                topics.extend([term for term, _ in common_terms])  # Add terms to topics list only

            # Calculate embeddings only for extracted topics
            if len(topics) > 0:
                embed = np.array(model.encode(topics))
                # Pairwise cosine similarity matrix
                n = len(topics)
                similarity_matrix = np.array(
                    [[1.0 - cosine(embed[i], embed[j]) for j in range(n)] for i in range(n)]
                ).tolist()

    # Render the HTML page
    return render_template_string(
        html_template,
        terms_data=terms_data,
        similarity_matrix=similarity_matrix,
        topics=topics
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)