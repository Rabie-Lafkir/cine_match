# 🎬 CineMatch Backend

This is the backend for CineMatch, a collaborative filtering movie recommendation system powered by Flask and the MovieLens 25M dataset.

---

## 💡 Features

- GET `/api/movies` – Paginated movie list with filters and sorting
- POST `/api/recommend` – Recommend top-10 movies based on 6+ user ratings
- GET `/api/health` – Simple health check
- Poster images from TMDB, cached in `movies_with_posters.csv`
- In-memory recommendation engine with fast response time

---

## 📁 Structure

```
backend/
├── app.py                    # Flask API
├── recommender.py            # Item-based collaborative filtering logic
├── data/ml-25m/              # MovieLens 25M ratings + metadata
├── .env                      # TMDB_API_KEY
└── static/posters/           # Placeholder image (optional)
```

---

## 🔧 Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask API
python app.py
```

The server runs at: `http://localhost:5000`

---

## 🌍 Environment Variables

Create a `.env` file at the root of the backend directory:

```
TMDB_API_KEY=your_tmdb_key_here
```

This key is used to fetch poster URLs during preprocessing.

---

## 🔄 API Endpoints

| Method | Route             | Description                               |
|--------|------------------|-------------------------------------------|
| GET    | `/api/movies`    | List paginated movies with filters        |
| POST   | `/api/recommend` | Recommend top 10 movies (from ratings)    |
| GET    | `/api/health`    | Returns `{ status: "ok" }`                |

---

## ⚙️ Notes

- Uses `movies_with_posters.csv` to avoid live TMDB lookups
- Caches metadata in-memory for performance
- Fully compatible with the frontend (see frontend README)

---

## 📸 Demo Preview

Example request:

```
GET /api/movies?page=1&genre=comedy&year=2010&sort=rating&order=desc
```

Returns a paginated list of filtered movie data.