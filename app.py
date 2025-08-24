from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Sample movie dataset
# ------------------------
data = {
    'title': ['Interstellar', 'Inception', 'The Dark Knight', 'Shutter Island', 'The Prestige', 'Gravity'],
    'genre': ['Sci-Fi Adventure Drama',
              'Sci-Fi Action Thriller',
              'Action Crime Drama',
              'Mystery Thriller Drama',
              'Drama Mystery Sci-Fi',
              'Sci-Fi Thriller Adventure']
}

df = pd.DataFrame(data)

# Vectorize genres
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])
cosine_sim = cosine_similarity(genre_matrix)

# Recommendation function
def recommend_movie(movie_name):
    # Convert input to lowercase for comparison
    movie_name = movie_name.lower()
    
    # Match with dataset (case-insensitive)
    if movie_name not in df['title'].str.lower().values:
        return ["❌ Movie not found in database. Try Interstellar, Inception, Gravity, etc."]

    # Get index of the given movie
    idx = df[df['title'].str.lower() == movie_name].index[0]

    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices].tolist()
# ------------------------
# Flask App
# ------------------------
app = Flask(__name__)

# Simple HTML template
html_template = """
<!doctype html>
<html>
<head>
    <title>Movie Recommendation System</title>
</head>
<body style="font-family: Arial; text-align:center; margin-top:50px;">
    <h1>🎬 Movie Recommendation System</h1>
    <form method="post">
        <label for="movie">Enter a movie name:</label><br><br>
        <input type="text" id="movie" name="movie" placeholder="e.g. Interstellar" required>
        <button type="submit">Get Recommendations</button>
    </form>
    {% if recommendations %}
        <h2>Recommended Movies for "{{ movie }}"</h2>
        <ul>
        {% for rec in recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    movie = ""
    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = recommend_movie(movie)
    return render_template_string(html_template, recommendations=recommendations, movie=movie)

if __name__ == "__main__":
    app.run(debug=True)
