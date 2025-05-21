import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ─── Config ─────────────────────────────────────────────────────
PLACEHOLDER_URL = "/static/posters/placeholder.jpg"

# ─── Recommender Class ──────────────────────────────────────────
class ItemCFRecommender:
    """
    Item-based collaborative filter using MovieLens-25M.
    Uses pre-fetched poster URLs from movies_with_posters.csv.
    """

    def __init__(self, root: str = "data/ml-25m") -> None:
        self.root = Path(root)

        print(f"⚙️  Loading ratings from {self.root}", flush=True)
        self.ratings = self._load_ratings()

        print("⚙️  Loading movie metadata with posters …", flush=True)
        self.movies = self._load_movies()

        print("⚙️  Building sparse user×item matrix …", flush=True)
        self.movie_ids = np.sort(self.ratings["movieId"].unique())
        self.id2idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.idx2id = {i: mid for mid, i in self.id2idx.items()}

        rows = self.ratings["userId"].astype("int32").values - 1
        cols = self.ratings["movieId"].map(self.id2idx).values
        vals = self.ratings["rating"].values.astype("float32")

        n_users = self.ratings["userId"].max()
        self.user_item = csr_matrix(
            (vals, (rows, cols)),
            shape=(n_users, len(self.movie_ids)),
            dtype=np.float32,
        )

        self.movies.set_index("movieId", inplace=True)
        self.avg_rating = (
            self.ratings.groupby("movieId")["rating"].mean().round(2)
        )
        print("✅ Recommender ready:",
              len(self.movie_ids), "movies |",
              n_users, "users", flush=True)

    def sample_movies(self, n: int = 150, seed: int = 42):
        rng = np.random.default_rng(seed)
        ids = rng.choice(self.movie_ids, size=n, replace=False)
        return self._meta_for(ids)

    def all_movies(self):
        return self._meta_for(self.movie_ids)

    def recommend_for_new_user(self, rating_list, top_n: int = 10,
                               min_rated: int = 6):
        if len(rating_list) < min_rated:
            raise ValueError(f"Need at least {min_rated} ratings")

        scores   = np.zeros(len(self.movie_ids), dtype=np.float32)
        sim_sums = np.zeros_like(scores)

        for pair in rating_list:
            mid, r = pair["movieId"], pair["rating"]
            if mid not in self.id2idx:
                continue
            idx = self.id2idx[mid]

            sims = cosine_similarity(
                self.user_item.T[idx], self.user_item.T
            ).flatten()

            scores   += sims * r
            sim_sums += np.abs(sims)

        with np.errstate(divide="ignore", invalid="ignore"):
            preds = np.divide(scores, sim_sums,
                              out=np.zeros_like(scores),
                              where=sim_sums != 0)

        rated_idx = {self.id2idx[p["movieId"]] for p in rating_list
                     if p["movieId"] in self.id2idx}
        preds[list(rated_idx)] = -np.inf

        top_idx = np.argsort(preds)[-top_n:][::-1]
        top_ids = [self.idx2id[i] for i in top_idx]
        return self._meta_for(top_ids, scores=preds[top_idx])

    def _load_ratings(self):
        path = self.root / "ratings.csv"
        use  = ["userId", "movieId", "rating"]
        dtyp = {"userId": "int32", "movieId": "int32", "rating": "float32"}
        return pd.read_csv(path, usecols=use, dtype=dtyp)

    def _load_movies(self):
        path = self.root / "movies_with_posters.csv"
        df = pd.read_csv(path)
        # Ensure no NaNs in critical fields
        df = df.fillna("")
        df = df[df["title"].str.strip() != ""]
        df = df[df["year"].astype(str).str.strip() != ""]
        df = df[df["genres"].str.strip() != ""]
        df = df[df["posterUrl"].str.startswith("http")]
        return df[["movieId", "title", "year", "genres", "posterUrl"]]

    def _meta_for(self, movie_ids, scores=None):
        rows: List[Dict] = []

        for i, mid in enumerate(movie_ids):
            if mid not in self.movies.index:
                continue

            meta = self.movies.loc[mid]
            item = {
                "movieId"  : int(mid),
                "title"    : meta["title"],
                "year"     : meta["year"],
                "genres"   : meta["genres"],
                "avgRating": float(self.avg_rating.get(mid, 0)),
                "posterUrl": meta["posterUrl"] or PLACEHOLDER_URL,
            }
            if scores is not None:
                item["score"] = float(scores[i])
            rows.append(item)

        return rows
