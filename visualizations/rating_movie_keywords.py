import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

# Load
movies = pd.read_csv("tmdb/tmdb_5000_movies.csv")

# Filter movies with 90th percentile popularity
top_rated_movies = movies[movies["vote_average"] > movies["vote_average"].quantile(0.90)]

# Function to extract 'name' values from JSON-like columns
def extract_names(json_str):
    try:
        items = ast.literal_eval(json_str)  # Convert string to list of dictionaries
        return [item["name"] for item in items]  # Extract only the 'name' values
    except (ValueError, SyntaxError):
        return []  # Return empty list if parsing fails

# Apply function to extract keywords and genres
top_rated_movies["keywords"] = top_rated_movies["keywords"].apply(extract_names)
top_rated_movies["genres"] = top_rated_movies["genres"].apply(extract_names)

# Combine genres and keywords into one list per movie
top_rated_movies["all_descriptors"] = top_rated_movies["keywords"] + top_rated_movies["genres"]

# Flatten the list of all descriptors
all_descriptors = [descriptor for sublist in top_rated_movies["all_descriptors"] for descriptor in sublist]

# Count the occurrences of each descriptor
descriptor_counts = Counter(all_descriptors)

# Get the top 15 descriptors
top_descriptors = descriptor_counts.most_common(15)
descriptors, counts = zip(*top_descriptors)

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(descriptors), palette="mako")

# Add labels and title
plt.xlabel("Count", fontsize=12)
plt.ylabel("Keyword / Genre", fontsize=12)
plt.title("Top Keywords and Genres in Top-Rated Movies", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.show()
