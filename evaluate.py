"""
Lightweight offline ranking evaluation.

For each eligible user, hold out their latest rating as a test interaction.
Build the recommender from the remaining ratings, seed it with the user's
highest-rated training movies, and calculate Precision@10 and Recall@10 for
held-out movies rated >= 4.0. This is a small development evaluation, not a
guarantee of real-world performance.
"""
import numpy as np
import pandas as pd
from src.recommender import load_source_data, fit_model, recommend

TOP_N = 10
MAX_USERS = 200  # keep evaluation fast on a normal laptop

def main():
    movies, ratings = load_source_data()
    ratings = ratings.sort_values("timestamp")
    eligible = ratings.groupby("userId").size()
    eligible_users = eligible[eligible >= 10].index.tolist()
    if len(eligible_users) > MAX_USERS:
        # Deterministic sample so results can be reproduced.
        eligible_users = sorted(eligible_users)[:MAX_USERS]

    train_parts, test_rows = [], []
    eligible_set = set(eligible_users)
    for user_id, group in ratings.groupby("userId"):
        if user_id not in eligible_set:
            train_parts.append(group)
            continue
        group = group.sort_values("timestamp")
        test_rows.append(group.iloc[-1])
        train_parts.append(group.iloc[:-1])

    train_ratings = pd.concat(train_parts, ignore_index=True)
    test = pd.DataFrame(test_rows)
    model = fit_model(movies, train_ratings)

    precisions, recalls, hit_rates = [], [], []
    evaluated = 0
    for row in test.itertuples(index=False):
        user_id = int(row.userId)
        true_movie = int(row.movieId)
        true_rating = float(row.rating)
        if true_rating < 4.0:
            continue

        user_train = train_ratings[train_ratings["userId"] == user_id]
        seeds = user_train[user_train["rating"] >= 4.0].sort_values(
            "rating", ascending=False
        )["movieId"].head(5).tolist()
        if not seeds:
            continue

        try:
            recs = recommend(model, seeds, top_n=TOP_N)
        except (ValueError, KeyError):
            continue

        recommended_ids = set(recs["movieId"].astype(int).tolist())
        hit = int(true_movie in recommended_ids)
        precisions.append(hit / TOP_N)
        recalls.append(float(hit))
        hit_rates.append(float(hit))
        evaluated += 1

    print("\n--- Offline Recommendation Evaluation ---")
    print(f"Users evaluated: {evaluated}")
    if evaluated:
        print(f"Precision@{TOP_N}: {np.mean(precisions):.4f}")
        print(f"Recall@{TOP_N}:    {np.mean(recalls):.4f}")
        print(f"Hit Rate@{TOP_N}:  {np.mean(hit_rates):.4f}")
    else:
        print("Not enough eligible held-out high ratings to calculate metrics.")
    print("Note: This is a small sampled holdout test. Interpret metrics carefully.")

if __name__ == "__main__":
    main()
