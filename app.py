from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import PureSVDRecommender
import time

app = Flask(__name__)
CORS(app)

# ── Initialize recommender ────────────────────────────
print("🔁 Initializing Pure SVD Recommender …")
rec = PureSVDRecommender(sample_users=5000, sample_movies=5000)
print("✅ PureSVD ready")


# ── Helper functions ──────────────────────────────────
def apply_filters(movies, genre_q, year_q, title_q):
    if title_q:
        movies = [m for m in movies if title_q in m["title"].lower()]
    if genre_q:
        movies = [m for m in movies if genre_q in m.get("genres", "").lower()]
    if year_q:
        try:
            movies = [m for m in movies if str(m["year"]).startswith(str(year_q))]
        except Exception:
            pass
    return movies

def apply_sorting(movies, sort_by, order):
    reverse = order == "desc"
    if sort_by == "rating":
        movies.sort(key=lambda m: m.get("avgRating", 0), reverse=reverse)
    elif sort_by == "year":
        movies.sort(key=lambda m: int(m.get("year", 0)), reverse=reverse)
    elif sort_by == "title":
        movies.sort(key=lambda m: m["title"].lower(), reverse=reverse)
    return movies

# ── Routes ─────────────────────────────────────────────
@app.route("/api/movies", methods=["GET"])
def get_movies():
    t0 = time.perf_counter()

    sample_param = request.args.get("sample", "150")
    page = max(int(request.args.get("page", 1)), 1)
    per_page = max(min(int(request.args.get("per_page", 30)), 200), 1)
    title_q = (request.args.get("search") or "").lower().strip()
    genre_q = (request.args.get("genre") or "").lower().strip()
    year_q = (request.args.get("year") or "").strip()
    sort_by = request.args.get("sort", "rating").strip().lower()
    order = request.args.get("order", "desc").strip().lower()

    # Get base movie list
    if sample_param == "all":
        movies = rec.all_movies()
    else:
        sample_size = int(sample_param)
        movies = rec.sample_movies(sample_size)

    # Apply filters and sorting
    movies = apply_filters(movies, genre_q, year_q, title_q)
    movies = apply_sorting(movies, sort_by, order)

    # Pagination
    total = len(movies)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = movies[start:end]

    return jsonify({
        "movies": page_items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "totalPages": (total + per_page - 1) // per_page,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)
    })

@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.get_json(force=True, silent=True) or {}
    rating_list = body.get("ratings", [])
    if not rating_list or len(rating_list) < 3:
        return jsonify({"error": "Need at least 3 ratings"}), 400
    recs = rec.recommend_for_new_user(rating_list, top_n=10)
    return jsonify({"recommended": recs})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
