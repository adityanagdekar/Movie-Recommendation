Movie Recommendation Project
# 🎬 Movie Recommendation System

A full-stack content-based movie recommendation system built using Python (FastAPI backend) and React.js (frontend).  
It recommends movies based on **movie title**, **movie overview**, **genres**, **cast**, **crew**, and **keywords** using **TF-IDF vectorization**, **cosine similarity**, and **Jaccard similarity filtering**.

---

## 🚀 Features

- 🔍 Content-based movie recommendations using textual metadata
- 🎯 Jaccard filtering on genres for thematic alignment
- 🔥 Popular movies section via demographic filtering (`vote_count`, `vote_average`, `popularity`)
- 🌐 FastAPI backend serving recommendations over HTTP
- ⚛️ React frontend with live search and modern UI

---

## 🧠 Recommendation Techniques

### ✅ 1. TF-IDF Vectorization
- Converts combined movie metadata into vectors
- Highlights important words (e.g., "wizard", "mission", "revenge") while down-weighting common ones

### ✅ 2. Cosine Similarity
- Measures angle-based similarity between movie vectors
- Finds movies that are textually close to the user input

### ✅ 3. Jaccard Evaluation
- Compares genres between the input and candidate movies
- Filters out movies with low thematic overlap (Jaccard < 0.2)

### ✅ 4. Demographic Filtering
- Ranks movies by a hybrid score:
  \[
  \text{score} = \frac{vote\_average \cdot vote\_count}{vote\_count + 500} + popularity
  \]

---

## 🏗 Tech Stack

| Layer       | Technology        |
|-------------|-------------------|
| Backend     | FastAPI (Python)  |
| ML/NLP      | scikit-learn, pandas, numpy |
| Frontend    | React.js (Vite)   |
| Data Source | TMDB Movie Dataset |

---
