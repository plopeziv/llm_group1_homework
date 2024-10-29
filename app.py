from flask import Flask, request, render_template_string
import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

# Download stopwords from nltk
nltk.download('stopwords')

app = Flask(__name__)

# Set of English stopwords
STOP_WORDS = set(stopwords.words('english'))

# HTML Template to render the form, summary terms, and table
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>CSV Upload and Filter</title>
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
    <h1>CSV File Data with Filter</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="csvfile">Upload CSV file:</label>
        <input type="file" name="csvfile" id="csvfile" accept=".csv">
        <br><br>
        <label for="sexFilter">Filter by Sex:</label>
        <br>
        <br>
        <input type="radio" id="all" name="sexFilter" value="ALL">
        <label for="all">ALL</label>
        </br>
        <input type="radio" id="male" name="sexFilter" value="MALE">
        <label for="male">MALE</label>
        <br>
        <input type="radio" id="female" name="sexFilter" value="FEMALE">
        <label for="female">FEMALE</label>
        <br>
        <br><br>
        <button type="submit">Upload and Filter</button>
    </form>

    {% if common_terms %}
    <h2>Most Common Phrases in Summary</h2>
    <table border="1">
        <thead>
            <tr>
                <th>Phrase</th>
                <th>Frequency</th>
            </tr>
        </thead>
        <tbody>
        {% for term, count in common_terms %}
            <tr>
                <td>{{ term }}</td>
                <td>{{ count }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% endif %}

    {% if table %}
    <h2>CSV Data Table</h2>
    {{ table | safe }}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    table = None
    common_terms = None
    if request.method == "POST":
        file = request.files.get("csvfile")
        sex_filter = request.form.get("sexFilter")  # Get the selected sex filter
        
        if file and file.filename.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)

            # Filter based on sex (lowercase 'sex' column)
            if sex_filter == "MALE":
                df = df[df['sex'] == "MALE"]
            elif sex_filter == "FEMALE":
                df = df[df['sex'] == "FEMALE"]

            # Extract most common phrases from the "summary" column (lowercase 'summary')
            if 'summary' in df.columns:
                # Join all the summary text
                all_summaries = " ".join(df['summary'].dropna())

                # Clean the text: remove punctuation and split into words
                words = re.findall(r'\b\w+\b', all_summaries.lower())

                # Remove stop words
                filtered_words = [word for word in words if word not in STOP_WORDS]

                # Generate 2-grams (bigrams) or 3-grams (trigrams)
                bigrams = ngrams(filtered_words, 2)  # Use 3 for trigrams
                bigram_counts = Counter(bigrams)

                # Get the top 10 most common phrases (as bigrams or trigrams)
                common_terms = bigram_counts.most_common(10)

                # Convert tuples of n-grams back to strings for display
                common_terms = [(" ".join(term), count) for term, count in common_terms]

            # Convert the DataFrame to an HTML table
            table = df.to_html(classes='table table-striped', index=False)
    
    # Render the HTML page, passing in the table and common terms if available
    return render_template_string(html_template, table=table, common_terms=common_terms)

if __name__ == "__main__":
    app.run(debug=True)