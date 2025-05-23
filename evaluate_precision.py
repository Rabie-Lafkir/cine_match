import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from recommender import PureSVDRecommender

K = 10
THRESHOLD = 4.0
SAMPLE_USERS = 1000

print("🔄 Loading PureSVD recommender...")
rec = PureSVDRecommender(sample_users=1000, sample_movies=1000)

ratings = rec.ratings
users = ratings['userId'].drop_duplicates().sample(n=SAMPLE_USERS, random_state=42)

precisions = []

print("⚙️  Running Precision@10 evaluation...")
for user_id in tqdm(users):
    user_ratings = ratings[ratings['userId'] == user_id]
    if len(user_ratings) < 6:
        continue

    train, test = train_test_split(user_ratings, test_size=0.4, random_state=42)
    profile = np.zeros(rec.item_factors.shape[1], dtype=np.float32)  # Latent dim
    for _, row in train.iterrows():
        mid, r = row["movieId"], row["rating"]
        if mid not in rec.id2idx_m:
            continue
        idx = rec.id2idx_m[mid]
        profile += rec.item_factors[idx] * r

    scores = rec.item_factors @ profile
    rated_indices = [rec.id2idx_m[m] for m in train["movieId"] if m in rec.id2idx_m]
    scores[rated_indices] = -np.inf

    top_indices = np.argsort(scores)[-K:][::-1]
    top_ids = {rec.idx2id_m[i] for i in top_indices}
    test_liked_ids = set(test[test["rating"] >= THRESHOLD]["movieId"])
    hits = top_ids & test_liked_ids
    precision = len(hits) / K
    precisions.append(precision)

mean_precision = np.mean(precisions)
print(f"\n✅ Precision@{K} = {mean_precision:.4f} based on {len(precisions)} users")
