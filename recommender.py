import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV
df = pd.read_csv("movies.csv")

# Combine genres and plot into a single feature
df["content"] = df["genres"].fillna('') + " " + df["plot"].fillna('')

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(df["content"])

# Calculate cosine similarity
similarity_matrix = cosine_similarity(vectors)

def recommend(movie_code, top_n=500):
    try:
        index = df[df["code"] == movie_code].index[0]
    except IndexError:
        return []

    scores = list(enumerate(similarity_matrix[index]))

    # Sort by similarity score (descending), skip the first (it will be itself)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Return top_n recommended movie codes
    recommended_codes = df.iloc[[i[0] for i in sorted_scores]]["code"].tolist()
    return recommended_codes
