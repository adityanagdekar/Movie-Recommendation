import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# loading data
movies = pd.read_csv("archive/tmdb_5000_movies.csv")
credits = pd.read_csv("archive/tmdb_5000_credits.csv")

# data pre-processing
movies = movies.merge(credits, on='title')

filtered_movies_df = movies[['movie_id', 'title', 'overview', 'keywords', 'genres', 'cast', 'crew', 'popularity', 'vote_average']]

filtered_movies_df.isnull().sum()

filtered_movies_df.dropna(inplace=True)

# handling missign values
filtered_movies_df.isnull().sum()

# checking duplicate values
filtered_movies_df.duplicated().sum()

# Create the MinMaxScaler scaler with output range (0, 100)
scaler = MinMaxScaler(feature_range=(0, 100))

popularity_scaled = scaler.fit_transform(filtered_movies_df[['popularity']])

filtered_movies_df['popularity_scaled'] = popularity_scaled

# Calculate the popularity cutoff for top 20%:
# this gives the value below which 80% of the data lies, i.e., 
# the top 20% movies by popularity are above this value.
high_pop_cutoff = filtered_movies_df['popularity_scaled'].quantile(0.80)
# print("popularity cutoff for top 20%: ", high_pop_cutoff)

# this gives the value below which 20% of the data lies, i.e., 
# the bottom 20% of movies.
low_pop_cutoff  = filtered_movies_df['popularity_scaled'].quantile(0.20)
# print("popularity cutoff below which 20% of the data lies: ", low_pop_cutoff)

# Create DataFrames for high & low popularity cutoffs
df_high_pop = filtered_movies_df[filtered_movies_df['popularity_scaled'] >= high_pop_cutoff]
df_low_pop  = filtered_movies_df[filtered_movies_df['popularity_scaled'] <= low_pop_cutoff]

def parse_keywords(keywords_data):
    try:
        keywords_list = []
        for keyword in ast.literal_eval(keywords_data):
            keywords_list.append(keyword['name'])
        return keywords_list
    except (ValueError, SyntaxError, TypeError):
        return [] # return empty list in case of any errors

filtered_movies_df['keywords'] = filtered_movies_df['keywords'].apply(parse_keywords)

filtered_movies_df['genres'] = filtered_movies_df['genres'].apply(parse_keywords)

def get_top_3_cast_members(cast_data):
    try:
        cast_members_list = []
        for cast_member in ast.literal_eval(cast_data):
            if (len(cast_members_list) <3):
                cast_members_list.append(cast_member['name'])
        return cast_members_list
    except (ValueError, SyntaxError, TypeError):
        return [] # return empty list in case of any errors

filtered_movies_df['cast'] = filtered_movies_df['cast'].apply(get_top_3_cast_members)

def get_director_producer(crew_data):
    try:
        crew_members_list = []
        for crew_member in ast.literal_eval(crew_data):
                if "job" in crew_member:
                    job_title = crew_member['job'].lower().strip()
                    if job_title in ["director", "producer"]:
                        crew_members_list.append(crew_member['name'])
        return list(set(crew_members_list))
    except (ValueError, SyntaxError, TypeError):
        return [] # return empty list in case of any errors

filtered_movies_df['crew'] = filtered_movies_df['crew'].apply(get_director_producer)

filtered_movies_df['overview'] = filtered_movies_df['overview'].apply(lambda x:x.split())

for col in ['keywords', 'genres', 'cast', 'crew']:
    filtered_movies_df[col] = filtered_movies_df[col].apply(
        lambda x:[i.replace(" ", "") for i in x] if isinstance(x, list) else []
    )

filtered_movies_df['movie_tag'] = filtered_movies_df['overview'] + filtered_movies_df['keywords'] + filtered_movies_df['genres'] + filtered_movies_df['cast'] + filtered_movies_df['crew']

filtered_movies_df['movie_tag'] = filtered_movies_df['movie_tag'].apply(lambda x: " ".join(x))

filtered_movies_df['movie_tag'] = filtered_movies_df['movie_tag'].apply(lambda x: x.lower())

filtered_movies_df["title"] = filtered_movies_df["title"].apply(lambda x: x.lower())

summarised_movies_df = filtered_movies_df[['movie_id', 'title', 'movie_tag', 'popularity_scaled', 'vote_average']]

summarised_movies_df["title"] = summarised_movies_df["title"].apply(lambda x: x.lower())

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ''

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize (split into words)
    tokens = text.split()

    # Remove stop words and apply stemming
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(cleaned)

summarised_movies_df['movie_tag'] = summarised_movies_df['movie_tag'].apply(preprocess_text)

# Assuming your DataFrame is called filtered_movies_df and contains a 'movie_tag' column
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Transform the 'movie_tag' column
tfidf_matrix = tfidf_vectorizer.fit_transform(summarised_movies_df['movie_tag'])

# tfidf_matrix is a sparse matrix of shape (n_movies, n_features)
print(tfidf_matrix.shape)

tfidf_vectorizer.get_feature_names_out()[:20]

similarity = cosine_similarity(tfidf_matrix)

# adding genres & keywords col.s from filtered_movies_df to summarised_movies_df
# summarised_movies_df[["genres", "keywords"]] = filtered_movies_df[["genres", "keywords"]] 
summarised_movies_df = summarised_movies_df.merge(
    filtered_movies_df[["movie_id", "genres", "keywords"]],
    on="movie_id",
    how="inner"   
)

# function to get movie id
def getMovieId(title):
    return summarised_movies_df.index[summarised_movies_df["title"] == title.lower().strip()][0]

def getRecommendationsList_Old(movie):
    # recommended movies list
    rec_movies_list = []

    # this list will store tuples -> (movie_id, similarity_val)
    enumerated_list = []

    # this list will store the idx of recommended movies
    rec_movies_idx_list = []

    # get movie idx
    matching_movies_df = summarised_movies_df[summarised_movies_df["title"].str.contains(movie)]
    if (len(matching_movies_df) > 0):
        for movie_idx in matching_movies_df.index:
            # get top-10 recommended movies

            # enumerate similairty with idx & store in list
            temp_enumerated_list = list(enumerate(similarity[movie_idx]))
            # extend the above list to the main enumerated_list
            enumerated_list.extend(temp_enumerated_list)

        # print("enumerated_list.len: ", len(enumerated_list))
        # sort the enumerated_list w.r.t 2nd ele. of the tuple which store the similarity
        sorted_enumerated_list = sorted(enumerated_list, reverse=True, key=lambda x: x[1])
        # print("top 10 enumerated_list items: ", sorted_enumerated_list[1:10])

        # and get top 20 results (leave the 0th tuple -> [its the show similarity of the movie with itself])
        rec_movies_idx_tuples = sorted_enumerated_list[1:20]
        # print(rec_movies_idx_tuples)
        # extract the movie ids from the list
        rec_movies_idx_list = list(rec_movies_idx_tuple[0] for rec_movies_idx_tuple in rec_movies_idx_tuples)
        # print("movie_ids: ", rec_movies_idx_list)

    return rec_movies_idx_list

def printRecommendations(movie_ids):
    movies_list = []
    # print the movie title 
    if (len(movie_ids) > 0):
        print("Top 20 recommended movies: ")  
        for id in movie_ids:
                print(f"{summarised_movies_df.iloc[id]['title']:50s} movie id: {id}")
                movies_list.append((id, summarised_movies_df.iloc[id]['title']))
    else:
        print("No matching movies found")
    return movies_list

def get_jaccard_score(user_inp_movie_id, rec_movie_id):
    i = user_inp_movie_id
    j = rec_movie_id

    user_inp_movie_genres = set(summarised_movies_df.iloc[i]["genres"])
    rec_movie_genres = set(summarised_movies_df.iloc[j]["genres"])

    score = len(user_inp_movie_genres & rec_movie_genres) / len(user_inp_movie_genres | rec_movie_genres)
    # if(score >0.2):
    #     print(f"{summarised_movies_df.iloc[j]['title']:50s}  score: {score} genres:{rec_movie_genres}")
    return score
    
def getRecommendations(title):
    title = title.lower().strip()

    # this list will store tuples -> (movie_id, similarity_val)
    enumerated_list = []

    # this list will store the idx of recommended movies
    filtered_rec_movies_idx_list = []

    # get movie idx
    matching_movies_df = summarised_movies_df[summarised_movies_df["title"].str.contains(title)]
    if (len(matching_movies_df) > 0):
        for movie_idx in matching_movies_df.index:
            # get top-10 recommended movies

            # enumerate similairty with idx & store in list
            temp_enumerated_list = list(enumerate(similarity[movie_idx]))
            # extend the above list to the main enumerated_list
            enumerated_list.extend(temp_enumerated_list)

        # print("enumerated_list.len: ", len(enumerated_list))
        # sort the enumerated_list w.r.t 2nd ele. of the tuple which store the similarity
        sorted_enumerated_list = sorted(enumerated_list, reverse=True, key=lambda x: x[1])
        # print("top 10 enumerated_list items: ", sorted_enumerated_list[1:10])

        # and get top 20 results (leave the 0th tuple -> [its the show similarity of the movie with itself])
        rec_movies_idx_tuples = sorted_enumerated_list[1:]

        # print(rec_movies_idx_tuples)
        # extract the movie ids from the list
        rec_movies_idx_list = list(rec_movies_idx_tuple[0] for rec_movies_idx_tuple in rec_movies_idx_tuples)
        # print("movie_ids: ", rec_movies_idx_list)
        # loop thru the list and get the top 10 recommended movies 
        for idx in rec_movies_idx_list:
            if (len(filtered_rec_movies_idx_list) < 20):
                movie_id = getMovieId(title)
                jaccard_score = get_jaccard_score(movie_id, idx)
                # if the thematic similarity is more than 0.3 then recommend it
                if jaccard_score > 0.2:
                    filtered_rec_movies_idx_list.append(idx)

    return filtered_rec_movies_idx_list

# def recommend_movies(title: str, top_k=10):
    

if __name__ == "__main__":
    #printRecommendations(getRecommendationsList_Old("spectre"))
    printRecommendations(getRecommendations("spectre"))