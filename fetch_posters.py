import os
import time
import argparse
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Load .env ──────────────────────────────────────────────
load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB_SEARCH = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_URL = "/static/posters/placeholder.jpg"
RATE_LIMIT_DELAY = 0.26

if not TMDB_KEY:
    raise SystemExit("❌ TMDB_API_KEY not set in environment or .env file")


def fetch_tmdb_poster(title: str, year: str) -> str:
    try:
        response = requests.get(
            TMDB_SEARCH,
            params={
                "api_key": TMDB_KEY,
                "query": title,
                "year": year,
                "include_adult": "false",
                "page": 1,
            },
            timeout=8
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        for movie in results:
            path = movie.get("poster_path")
            if path and isinstance(path, str) and path.lower() != "null":
                return TMDB_IMAGE_BASE + path
    except requests.RequestException as e:
        print(f"⚠️ TMDB error: {title} ({year}) → {e}")
    return PLACEHOLDER_URL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ml-25m/movies.csv", help="Original MovieLens movies.csv")
    parser.add_argument("--output", default="data/ml-25m/movies_with_posters.csv", help="Path to save results")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # ─── Load original MovieLens movies.csv ──────────────────────
    df_all = pd.read_csv(input_path)
    df_all[["title_clean", "year"]] = df_all["title"].str.extract(r"^(.*) \((\d{4})\)$")
    df_all["year"] = df_all["year"].fillna("")
    df_all["title_clean"] = df_all["title_clean"].fillna("")

    # ─── Resume support: check for existing output ───────────────
    if output_path.exists():
        df_done = pd.read_csv(output_path)
        done_ids = set(df_done["movieId"])
        print(f"🔁 Resuming from previous file. {len(done_ids)} already fetched.")
    else:
        df_done = pd.DataFrame()
        done_ids = set()

    # ─── Filter out already processed movies ─────────────────────
    df_remaining = df_all[~df_all["movieId"].isin(done_ids)].copy()
    print(f"📦 Fetching posters for {len(df_remaining)} remaining movies…")

    # ─── Main fetch loop ─────────────────────────────────────────
    poster_rows = []
    for _, row in tqdm(df_remaining.iterrows(), total=len(df_remaining)):
        movie_id = int(row["movieId"])
        title = row["title_clean"].strip()
        year = row["year"].strip()

        if not title or not year:
            poster_url = PLACEHOLDER_URL
        else:
            poster_url = fetch_tmdb_poster(title, year)
            time.sleep(RATE_LIMIT_DELAY)

        new_row = row.drop(labels=["title_clean"]).copy()
        new_row["posterUrl"] = poster_url
        poster_rows.append(new_row)

        # Write batch every 100 movies to avoid data loss
        if len(poster_rows) % 100 == 0:
            batch_df = pd.DataFrame(poster_rows)
            df_done = pd.concat([df_done, batch_df], ignore_index=True)
            df_done.to_csv(output_path, index=False)
            poster_rows = []

    # ─── Final write ─────────────────────────────────────────────
    if poster_rows:
        batch_df = pd.DataFrame(poster_rows)
        df_done = pd.concat([df_done, batch_df], ignore_index=True)
        df_done.to_csv(output_path, index=False)

    print(f"✅ Complete! Saved to: {args.output}")


if __name__ == "__main__":
    main()
