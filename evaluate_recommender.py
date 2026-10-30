
"""
Compare the original single-seed recommender with an improved
multiple-seed genre-based recommender.

Run:
    python evaluate_recommender.py

Required files:
    data/movie_recommendation_dataset.xlsx
    data/ratings.csv

Metrics:
    Precision@10
    Recall@10
    F1@10
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

MOVIE_FILE = DATA / "movie_recommendation_dataset.xlsx"
RATINGS_FILE = DATA / "ratings.csv"

RANDOM_SEED = 42
TEST_FRACTION = 0.20
RELEVANT_THRESHOLD = 4.0
TOP_N = 10
MIN_USER_RATINGS = 5


def stop(message):
    print("\nERROR:", message)
    sys.exit(1)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

for file in (MOVIE_FILE, RATINGS_FILE):
    if not file.exists():
        stop(f"Missing file: {file}")

try:
    movies = pd.read_excel(
        MOVIE_FILE,
        sheet_name="Movies"
    )
except Exception as exc:
    stop(f"Could not read Excel file: {exc}")

ratings = pd.read_csv(RATINGS_FILE)

required_movie_cols = {
    "movieId", "MovieTitle", "Genre"
}

required_rating_cols = {
    "userId", "movieId", "rating"
}

if not required_movie_cols.issubset(movies.columns):
    stop("Required movie columns are missing.")

if not required_rating_cols.issubset(ratings.columns):
    stop("Required rating columns are missing.")


# --------------------------------------------------
# 3. CLEAN DATA
# --------------------------------------------------

movies = movies.dropna(
    subset=["movieId", "MovieTitle", "Genre"]
).copy()

movies["movieId"] = pd.to_numeric(
    movies["movieId"], errors="coerce"
)

movies = movies.dropna(
    subset=["movieId"]
)

movies["movieId"] = movies["movieId"].astype(int)

movies["MovieTitle"] = (
    movies["MovieTitle"].astype(str).str.strip()
)

movies["Genre"] = (
    movies["Genre"].astype(str).str.strip()
)

movies = movies[
    (movies["MovieTitle"] != "")
    & (movies["Genre"] != "")
]

movies = movies.drop_duplicates(
    "movieId"
).reset_index(drop=True)


ratings["userId"] = pd.to_numeric(
    ratings["userId"], errors="coerce"
)

ratings["movieId"] = pd.to_numeric(
    ratings["movieId"], errors="coerce"
)

ratings["rating"] = pd.to_numeric(
    ratings["rating"], errors="coerce"
)

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
).copy()

ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

valid_movie_ids = set(movies["movieId"])

ratings = ratings[
    ratings["movieId"].isin(valid_movie_ids)
].copy()

if ratings.empty:
    stop("No matching movie IDs were found.")


# --------------------------------------------------
# 4. BUILD TF-IDF GENRE MATRIX
# --------------------------------------------------

def genre_tokenizer(text):
    return [
        genre.strip().lower()
        for genre in str(text).split(",")
        if genre.strip()
    ]


vectorizer = TfidfVectorizer(
    tokenizer=genre_tokenizer,
    token_pattern=None,
    lowercase=False,
    norm="l2"
)

genre_matrix = vectorizer.fit_transform(
    movies["Genre"]
)

movie_ids = movies["movieId"].tolist()

id_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

movie_titles = movies["MovieTitle"].tolist()


# --------------------------------------------------
# 5. RECOMMENDATION FUNCTION
# --------------------------------------------------

def get_recommendations(seed_indices, candidate_indices):

    similarity_scores = cosine_similarity(
        genre_matrix[seed_indices],
        genre_matrix[candidate_indices]
    )

    # Average similarity to all selected favourite movies.
    average_scores = similarity_scores.mean(axis=0)

    ranked = sorted(
        zip(candidate_indices, average_scores),
        key=lambda item: (
            -float(item[1]),
            movie_titles[item[0]].casefold()
        )
    )

    return [
        int(movie_ids[index])
        for index, score in ranked[:TOP_N]
    ]


# --------------------------------------------------
# 6. CALCULATE EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(recommended_ids, relevant_ids):

    hits = len(
        set(recommended_ids) & relevant_ids
    )

    precision = (
        hits / len(recommended_ids)
        if recommended_ids else 0.0
    )

    recall = (
        hits / len(relevant_ids)
        if relevant_ids else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1


# --------------------------------------------------
# 7. USER-LEVEL TRAIN/TEST EVALUATION
# --------------------------------------------------

results_original = []
results_improved = []

skipped_no_relevant = 0
skipped_no_seed = 0
skipped_no_candidates = 0

for user_id, group in ratings.groupby("userId"):

    group = group.sample(
        frac=1,
        random_state=RANDOM_SEED + int(user_id)
    ).reset_index(drop=True)

    if len(group) < MIN_USER_RATINGS:
        continue

    test_size = max(
        1,
        int(round(len(group) * TEST_FRACTION))
    )

    test_size = min(
        test_size,
        len(group) - 1
    )

    test = group.iloc[:test_size]
    train = group.iloc[test_size:]

    # Movies liked in the held-out test set.
    relevant_ids = set(
        test.loc[
            test["rating"] >= RELEVANT_THRESHOLD,
            "movieId"
        ].astype(int)
    )

    if not relevant_ids:
        skipped_no_relevant += 1
        continue

    # Movies liked in the training set.
    positive_train_ids = train.loc[
        train["rating"] >= RELEVANT_THRESHOLD,
        "movieId"
    ].astype(int).tolist()

    positive_train_ids = [
        mid for mid in positive_train_ids
        if mid in id_to_index
    ]

    if not positive_train_ids:
        skipped_no_seed += 1
        continue

    seed_indices = [
        id_to_index[mid]
        for mid in positive_train_ids
    ]

    # Exclude movies already rated in training.
    seen_ids = set(
        train["movieId"].astype(int)
    )

    candidate_indices = [
        index
        for index, mid in enumerate(movie_ids)
        if mid not in seen_ids
    ]

    if not candidate_indices:
        skipped_no_candidates += 1
        continue

    # ----------------------------------------------
    # ORIGINAL METHOD: ONE RANDOM FAVOURITE MOVIE
    # ----------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED + int(user_id)
    )

    selected_seed = seed_indices[
        int(rng.integers(0, len(seed_indices)))
    ]

    original_recommendations = get_recommendations(
        [selected_seed],
        candidate_indices
    )

    # ----------------------------------------------
    # IMPROVED METHOD: MULTIPLE FAVOURITE MOVIES
    # ----------------------------------------------

    improved_recommendations = get_recommendations(
        seed_indices,
        candidate_indices
    )

    # Calculate both methods against the same test set.
    original_metrics = calculate_metrics(
        original_recommendations,
        relevant_ids
    )

    improved_metrics = calculate_metrics(
        improved_recommendations,
        relevant_ids
    )

    results_original.append(original_metrics)
    results_improved.append(improved_metrics)


# --------------------------------------------------
# 8. DISPLAY FINAL RESULTS
# --------------------------------------------------

if not results_original:
    stop(
        "No evaluable users found. Check your ratings.csv "
        "and matching movie IDs."
    )

original_array = np.asarray(
    results_original,
    dtype=float
)

improved_array = np.asarray(
    results_improved,
    dtype=float
)

original_average = original_array.mean(axis=0)
improved_average = improved_array.mean(axis=0)

print("\n==============================================")
print(" MOVIE RECOMMENDATION SYSTEM EVALUATION")
print("==============================================")

print("Model: Genre TF-IDF + Cosine Similarity")

print(
    f"Relevance threshold: {RELEVANT_THRESHOLD}/5"
)

print(f"Top recommendations: {TOP_N}")
print(f"Test fraction: {TEST_FRACTION:.0%}")

print(
    f"Users evaluated: {len(results_original)}"
)

print(
    f"Users skipped (no relevant test movies): "
    f"{skipped_no_relevant}"
)

print(
    f"Users skipped (no liked training movies): "
    f"{skipped_no_seed}"
)

print(
    f"Users skipped (no candidates): "
    f"{skipped_no_candidates}"
)

print("\n----------------------------------------------")
print(" COMPARISON OF EVALUATION RESULTS")
print("----------------------------------------------")

print(
    f"{'Metric':<16}"
    f"{'Original':>15}"
    f"{'Improved':>15}"
)

print("-" * 46)

metric_names = [
    "Precision@10",
    "Recall@10",
    "F1@10"
]

for index, name in enumerate(metric_names):

    old_value = original_average[index] * 100
    new_value = improved_average[index] * 100

    print(
        f"{name:<16}"
        f"{old_value:>14.2f}%"
        f"{new_value:>14.2f}%"
    )

print("\n==============================================")
print(" EVALUATION COMPLETED")
print("==============================================")

print("\nImportant notes:")
print("- Results are macro-averaged across evaluated users.")
print("- Both methods use identical train/test splits.")
print("- Improved recommendations use multiple liked movies.")
print("- Higher scores are not guaranteed.")
print("- These are recommendation metrics, not accuracy.")
print("- The system still uses genre-only content information.")
