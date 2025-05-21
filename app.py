"""
CineMatch Flask API – MovieLens-25M (filters + sorting, fixed year issue)
"""

from functools import lru_cache
from typing import List, Dict, Tuple
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

from recommender import ItemCFRecommender

# ── App setup ───────────────────────────
app = Flask(__name__)
CORS(app, origins="*")
rec_engine = ItemCFRecommender(root="data/ml-25m")

# ── Movie cache ─────────────────────────
ALL_MOVIES: List[Dict] = []
SAMPLE_CACHE: Dict[int, List[Dict]] = {}

def get_sample(sample: str) -> List[Dict]:
    global ALL_MOVIES
    if sample == "all":
        if not ALL_MOVIES:
            print("📦 Caching full movie list …", flush=True)
            ALL_MOVIES = rec_engine.all_movies()
            print(f"✅ {len(ALL_MOVIES)} movies cached", flush=True)
        return ALL_MOVIES
    k = int(sample)
    if k not in SAMPLE_CACHE:
        SAMPLE_CACHE[k] = rec_engine.sample_movies(k)
    return SAMPLE_CACHE[k]

@lru_cache(maxsize=1024)
def slice_page(
    sample: str, title_q: str, genre_q: str, year_q: str,
    sort_by: str, order: str,
    page: int, per_page: int
) -> Tuple[List[Dict], int]:
    movies = get_sample(sample)

    # ── Filters ─────────────────────────────
    if title_q:
        movies = [m for m in movies if title_q in m["title"].lower()]
    if genre_q:
        movies = [m for m in movies if genre_q in m.get("genres", "").lower()]
    if year_q:
        def safe_year_filter(m):
            try:
                y = int(float(m["year"]))
                if year_q.lower() == "older":
                    return y < 2000
                elif year_q.isdigit():
                    return y == int(year_q)
                elif year_q.startswith(">="):
                    return y >= int(year_q[2:])
                elif year_q.startswith("<="):
                    return y <= int(year_q[2:])
            except:
                return False
            return False
        movies = [m for m in movies if safe_year_filter(m)]

    # ── Sorting ─────────────────────────────
    reverse = order == "desc"
    if sort_by == "rating":
        movies.sort(key=lambda m: m.get("avgRating", 0), reverse=reverse)
    elif sort_by == "year":
        def safe_year(m):
            try:
                return int(float(m.get("year", 0)))
            except:
                return 0
        movies.sort(key=safe_year, reverse=reverse)
    elif sort_by == "title":
        movies.sort(key=lambda m: m["title"].lower(), reverse=reverse)

    # ── Pagination ──────────────────────────
    total = len(movies)
    start = (page - 1) * per_page
    end = start + per_page
    return movies[start:end], total

# ── Routes ────────────────────────────────
@app.get("/api/movies")
def movies_route():
    t0 = time.perf_counter()

    # Params
    sample    = request.args.get("sample", "all")
    page      = max(int(request.args.get("page", 1)), 1)
    per_page  = max(min(int(request.args.get("per_page", 30)), 200), 1)
    title_q   = (request.args.get("search") or "").lower().strip()
    genre_q   = (request.args.get("genre") or "").lower().strip()
    year_q    = (request.args.get("year") or "").strip()
    sort_by   = request.args.get("sort", "rating").strip().lower()
    order     = request.args.get("order", "desc").strip().lower()

    print(f"📥 /api/movies → page={page} genre={genre_q} year={year_q} sort={sort_by}/{order}", flush=True)

    page_items, total = slice_page(
        sample, title_q, genre_q, year_q, sort_by, order, page, per_page
    )

    return jsonify({
        "movies": page_items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "totalPages": (total + per_page - 1) // per_page,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)
    })

@app.post("/api/recommend")
def recommend():
    body = request.get_json(force=True, silent=True) or {}
    rating_list = body.get("ratings", [])
    recs = rec_engine.recommend_for_new_user(rating_list, 10, 6)
    return jsonify({"recommended": recs})

@app.get("/api/health")
def health():
    return {"status": "ok"}

# ── Run ───────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
