import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

PLACEHOLDER_URL = "/static/posters/placeholder.jpg"

class PureSVDRecommender:
    def __init__(self, root: str = "data/ml-25m", sample_users=5000, sample_movies=5000):
        self.root = Path(root)
        print("⚙️  Loading ratings and movies ...")
        self.ratings = self._load_ratings(sample_users, sample_movies)
        self.movies = self._load_movies()
        self.movie_ids = sorted(self.ratings["movieId"].unique())
        self.user_ids = sorted(self.ratings["userId"].unique())

        self.id2idx_m = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.idx2id_m = {i: mid for mid, i in self.id2idx_m.items()}
        self.id2idx_u = {uid: i for i, uid in enumerate(self.user_ids)}

        print("⚙️  Creating user-item matrix ...")
        rows = self.ratings["userId"].map(self.id2idx_u).values
        cols = self.ratings["movieId"].map(self.id2idx_m).values
        data = self.ratings["rating"].values.astype(np.float32)

        self.matrix = csr_matrix((data, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids)))
        print("⚙️  Computing SVD ...")
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.user_factors = self.svd.fit_transform(self.matrix)
        self.item_factors = self.svd.components_.T
        print("✅ PureSVD ready:", self.matrix.shape)

        self.movies.set_index("movieId", inplace=True)
        self.avg_rating = self.ratings.groupby("movieId")["rating"].mean().round(2)

    def recommend_for_new_user(self, rating_list, top_n=10, min_rated=6):
        if len(rating_list) < min_rated:
            raise ValueError(f"Need at least {min_rated} ratings")

        user_profile = np.zeros(self.item_factors.shape[1])
        for r in rating_list:
            mid, rating = r["movieId"], r["rating"]
            if mid not in self.id2idx_m:
                continue
            idx = self.id2idx_m[mid]
            user_profile += self.item_factors[idx] * rating

        scores = self.item_factors @ user_profile
        rated_indices = [self.id2idx_m[r["movieId"]] for r in rating_list if r["movieId"] in self.id2idx_m]
        scores[rated_indices] = -np.inf

        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_ids = [self.idx2id_m[i] for i in top_indices]
        return self._meta_for(top_ids, scores=scores[top_indices])

    def sample_movies(self, n: int = 150, seed: int = 42):
        rng = np.random.default_rng(seed)
        ids = rng.choice(self.movie_ids, size=n, replace=False)
        return self._meta_for(ids)

    def all_movies(self):
        return self._meta_for(self.movie_ids)

    def _load_ratings(self, max_users, max_movies):
        df = pd.read_csv(self.root / "ratings.csv", usecols=["userId", "movieId", "rating"])
        users = df["userId"].unique()[:max_users]
        df = df[df["userId"].isin(users)]
        movies = df["movieId"].unique()[:max_movies]
        return df[df["movieId"].isin(movies)]

    def _load_movies(self):
        df = pd.read_csv(self.root / "movies_with_posters.csv")
        df = df.fillna("")
        df = df[df["title"].str.strip() != ""]
        df = df[df["year"].astype(str).str.strip() != ""]
        df = df[df["genres"].str.strip() != ""]
        df = df[df["posterUrl"].str.startswith("http")]
        return df[["movieId", "title", "year", "genres", "posterUrl"]]

    def _meta_for(self, movie_ids, scores=None):
        rows = []
        for i, mid in enumerate(movie_ids):
            if mid not in self.movies.index:
                continue
            m = self.movies.loc[mid]
            item = {
                "movieId": int(mid),
                "title": m["title"],
                "year": m["year"],
                "genres": m["genres"],
                "avgRating": float(self.avg_rating.get(mid, 0)),
                "posterUrl": m["posterUrl"] or PLACEHOLDER_URL,
            }
            if scores is not None:
                item["score"] = float(scores[i])
            rows.append(item)
        return rows
