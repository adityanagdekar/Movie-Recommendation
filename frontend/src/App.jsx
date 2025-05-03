import React, { useState, useEffect } from 'react';

function App() {
  const [title, setTitle] = useState("");
  const [movies, setMovies] = useState([]);
  const [popularMovies, setPopularMovies] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/popular")
      .then((res) => res.json())
      .then((data) => setPopularMovies(data.results || []));
  }, []);

  const fetchRecommendations = async () => {
    const res = await fetch(`http://127.0.0.1:8000/recommend?title=${encodeURIComponent(title)}`);
    const data = await res.json();
    setMovies(data.results || []);
  };

  return (
    <div className='main-container'>
      <h1>🎬 Movie Recommender</h1>

      <div className='input-container'>
        <input
          type="text"
          value={title}
          placeholder="Enter a movie title"
          onChange={(e) => setTitle(e.target.value)}
          // style={{ padding: "0.5rem", width: "300px" }}
        />
        <button onClick={fetchRecommendations} style={{ marginLeft: "1rem" }}>
          Recommend
        </button>
      </div>
      
      <div className='movies-section'>
        <div className='recommended-movies-section'>
            <h3>Recommended movies</h3>
            {movies.length > 0 ? (
              <ul>
              {movies.map((movie, i) => (
                <li key={i}>
                  <strong>{movie[1]}</strong>
                </li>
              ))}
            </ul>
            ) : (
              <p>No recommendations found. Try another movie title.</p>
            )}
          </div>

          <div className='popular-movies-section'>
            <h3>Popular Right Now 🔥</h3>
            <ul>
              {popularMovies.map((movie, i) => (
                <li key={i}><strong>{movie[1]}</strong></li>
              ))}
            </ul>
          </div>
      </div>
    </div>
  );
}

export default App;
