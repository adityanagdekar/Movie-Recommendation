# prepare data
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load and merge
movies = pd.read_csv("tmdb/tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb/tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')

# Parse JSON-like fields
def parse_names(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)]
    except:
        return []

movies['genres'] = movies['genres'].apply(parse_names)
movies['keywords'] = movies['keywords'].apply(parse_names)
movies['production_companies'] = movies['production_companies'].apply(parse_names)

def get_lead_cast(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)[:3]]
    except:
        return []

movies['cast'] = movies['cast'].apply(get_lead_cast)

def get_director(x):
    try:
        crew_list = ast.literal_eval(x)
        for member in crew_list:
            if member['job'] == 'Director':
                return member['name']
        return ''
    except:
        return ''

movies['director'] = movies['crew'].apply(get_director)

def collapse_features(row):
    return ' '.join(row['genres']) + ' ' + ' '.join(row['keywords']) + ' ' + ' '.join(row['cast']) + ' ' + row['director'] + ' ' + ' '.join(row['production_companies'])

movies['tags'] = movies.apply(collapse_features, axis=1)

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Add normalized vote and popularity
movies['norm_vote'] = movies['vote_average'] / 10
movies['log_popularity'] = np.log1p(movies['popularity'])  # avoids log(0)

# User input to imitate real-world scenario
user_ratings = {
    "Inception": 4,
    "Avatar": 2,
    "Harry Potter and the Half-Blood Prince": 2,
    "Cars 2": 4,
    "Ant-Man": 1,
    "Finding Nemo": 4,
    "The Hunger Games: Mockingjay - Part 2": 3
}

# Updated recommendation function
def recommend(user_ratings, similarity, movies_df):
    scores = np.zeros(len(movies_df))

    for title, rating in user_ratings.items():
        idx = movies_df[movies_df['title'] == title].index
        if len(idx) == 0:
            continue
        sim_scores = similarity[idx[0]]
        adjusted_rating = rating - 3  # center around neutral
        scores += sim_scores * adjusted_rating

    # Normalize similarity scores to [0, 1]
    sim_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    # Normalize log_popularity to [0, 1]
    pop_norm = movies_df['log_popularity'] / movies_df['log_popularity'].max()

    # Final score = weighted sum
    movies_df['final_score'] = (
        0.6 * sim_norm +
        0.25 * movies_df['norm_vote'] +
        0.15 * pop_norm
    )

    # Remove user-rated movies
    unrated = movies_df[~movies_df['title'].isin(user_ratings.keys())]

    # Recommend top 10
    return unrated.sort_values('final_score', ascending=False).head(10)

# Get recommendations
recs = recommend(user_ratings, similarity, movies)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=recs, x='popularity', y='vote_average', s=100, color='dodgerblue')

# Label each point
for i in range(len(recs)):
    plt.text(
        recs.iloc[i]['popularity'] + 0.5,
        recs.iloc[i]['vote_average'],
        recs.iloc[i]['title'],
        fontsize=9,
        ha='left'
    )

plt.title("Recommendation Plot: Vote Average vs Popularity")
plt.xlabel("Popularity")
plt.ylabel("Rating")
plt.grid(True)
plt.tight_layout()
plt.show()
