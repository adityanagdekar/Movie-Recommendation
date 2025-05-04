
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
## 📁 Folder Structure

```text
movie-recommender/
├── backend/
│   ├── recommender_api.py         # FastAPI server
│   ├── recommender_module.py      # Core logic for recommendations
│   ├── rec_system.py              # Possibly separate logic (Word2Vec, TF-IDF, etc.)
│   ├── requirements.txt           # 🔥 Environment dependencies
│   ├── movie-recommender.ipynb    # Original prototype
│   └── archive/                   # Older/notebook artifacts
├── frontend/
│   ├── src/                       # React components
│   ├── public/
│   └── package.json               # React dependencies
├── Presentations/                 # Slide deck, reports
├── visualizations/                # Charts, graphs
└── README.md                      # 🔥 Root readme for GitHub
```

---
## 🔧 How to Run

### 🐍 Backend (FastAPI)
```text
cd backend
conda create -n movie-recs python=3.10
conda activate movie-recs
pip install -r requirements.txt
uvicorn recommender_api:app --reload
```

---
### ⚛️ Frontend (React)
```text
cd frontend
npm install
npm run dev
```

---
🖼 Sample Demo
✏️ Input: "Jack"

🎬 Recommended: "Jack the giant slayer", "Percy Jackson: sea of monsters", "mighty joe young", etc.

🔥 Popular Movies: Shown via separate carousel on the page

---
📌 Future Enhancements
✅ Add Word2Vec or BERT for semantic recommendations

⏳ Integrate collaborative filtering with synthetic user profiles

🌍 Deploy to Vercel (frontend) & Render (backend)

---
🙌 Acknowledgements
TMDB for the dataset

scikit-learn for TF-IDF and similarity

FastAPI and Vite for lightning-fast dev experience
