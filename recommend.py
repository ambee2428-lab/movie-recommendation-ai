from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path("data/movie_recommendation_dataset.xlsx")
SHEET_NAME = "Movies"
TOP_N = 10

REQUIRED_COLUMNS = ["MovieTitle", "ReleaseYear", "Genre", "Rating_10"]


def load_movies():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Make sure the Excel file is inside the project's data folder "
            "and is named movie_recommendation_dataset.xlsx."
        )

    movies = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

    missing = [column for column in REQUIRED_COLUMNS if column not in movies.columns]
    if missing:
        raise ValueError(
            "The Excel file is missing required column(s): "
            + ", ".join(missing)
        )

    movies = movies.copy()
    movies["MovieTitle"] = movies["MovieTitle"].fillna("").astype(str).str.strip()
    movies["Genre"] = movies["Genre"].fillna("").astype(str).str.strip()
    movies["ReleaseYear"] = pd.to_numeric(movies["ReleaseYear"], errors="coerce")
    movies["Rating_10"] = pd.to_numeric(movies["Rating_10"], errors="coerce")
    movies["Rating"] = movies["Rating_10"] / 2.0

    # A movie needs a title and at least one genre to calculate genre similarity.
    movies = movies[
        (movies["MovieTitle"] != "")
        & (movies["Genre"] != "")
        & movies["Rating"].notna()
    ].copy()

    movies.reset_index(drop=True, inplace=True)
    if movies.empty:
        raise ValueError("No usable movies were found in the Excel worksheet.")

    return movies


def build_similarity(movies):
    # Genre is stored as comma-separated labels. Treat each complete genre label
    # as one token (for example, "sci-fi"), rather than splitting it into words.
    def genre_tokenizer(text):
        return [genre.strip().lower() for genre in str(text).split(",") if genre.strip()]

    vectorizer = TfidfVectorizer(
        tokenizer=genre_tokenizer,
        token_pattern=None,
        lowercase=False,
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(movies["Genre"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


def get_year_filter(prompt):
    raw = input(prompt).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print("Please enter a whole year, or leave it blank.")
        return get_year_filter(prompt)


def recommend(movies, similarity_matrix):
    print("\n=== Movie Recommendation System (TF-IDF + Cosine Similarity) ===")
    print(f"Movies available for recommendations: {len(movies):,}")
    query = input("\nEnter a movie title you like: ").strip()

    if not query:
        print("Please enter a movie title.")
        return

    exact_matches = movies.index[
        movies["MovieTitle"].str.casefold() == query.casefold()
    ].tolist()

    if not exact_matches:
        title_matches = movies[
            movies["MovieTitle"].str.contains(query, case=False, regex=False, na=False)
        ]["MovieTitle"].drop_duplicates().head(10).tolist()

        if title_matches:
            print("\nI couldn't find an exact title. Possible matches:")
            for title in title_matches:
                print(f" - {title}")
            print("\nRun the program again and enter one of the titles above.")
        else:
            print("No matching title found. Check the spelling and try again.")
        return

    # If a title appears more than once, use its first matching dataset row.
    movie_index = exact_matches[0]
    selected = movies.loc[movie_index]

    print(f"\nSelected movie: {selected['MovieTitle']}")
    print(f"Genres: {selected['Genre']}")
    print(
        "Average rating: "
        + (f"{selected['Rating']:.2f}/5" if pd.notna(selected["Rating"]) else "N/A")
    )

    minimum_rating = input(
        "\nMinimum recommended movie rating (0 to 5, Enter for 0): "
    ).strip()
    if minimum_rating:
        try:
            minimum_rating = float(minimum_rating)
            if not 0 <= minimum_rating <= 5:
                raise ValueError
        except ValueError:
            print("Invalid rating. Enter a number from 0 to 5.")
            return
    else:
        minimum_rating = 0.0

    year_from = get_year_filter("Release year FROM (Enter for any year): ")
    year_to = get_year_filter("Release year TO (Enter for any year): ")

    if year_from is not None and year_to is not None and year_from > year_to:
        print("The starting year cannot be later than the ending year.")
        return

    scores = list(enumerate(similarity_matrix[movie_index]))
    scores = [
        (index, score)
        for index, score in scores
        if index != movie_index
    ]

    # Apply optional rating and year filters before displaying recommendations.
    candidates = []
    for index, score in scores:
        row = movies.loc[index]

        if row["Rating"] < minimum_rating:
            continue
        if year_from is not None:
            if pd.isna(row["ReleaseYear"]) or row["ReleaseYear"] < year_from:
                continue
        if year_to is not None:
            if pd.isna(row["ReleaseYear"]) or row["ReleaseYear"] > year_to:
                continue

        candidates.append((index, float(score)))

    # Similarity is the primary ranking criterion; average rating breaks ties.
    candidates.sort(
        key=lambda item: (
            item[1],
            float(movies.loc[item[0], "Rating"])
        ),
        reverse=True,
    )

    if not candidates:
        print("\nNo movies matched those filters. Try a lower rating or wider year range.")
        return

    print(f"\nTop {min(TOP_N, len(candidates))} recommendations:")
    print("-" * 92)
    print(f"{'Movie Title':48} {'Year':6} {'Rating / 5':12} {'Similarity':10}")
    print("-" * 92)

    for index, score in candidates[:TOP_N]:
        row = movies.loc[index]
        year = str(int(row["ReleaseYear"])) if pd.notna(row["ReleaseYear"]) else "N/A"
        print(
            f"{row['MovieTitle'][:47]:48} "
            f"{year:6} "
            f"{row['Rating']:>8.2f}/5  "
            f"{score:>9.3f}"
        )

    print("\nNote: Similarity scores range from 0 to 1. They are not accuracy percentages.")
    print(
        "This dataset contains genre metadata, so recommendations are based on genre "
        "similarity, not plot, director, cast, or personal viewing history."
    )


def main():
    try:
        movies = load_movies()
        similarity_matrix = build_similarity(movies)
        recommend(movies, similarity_matrix)
    except FileNotFoundError as error:
        print(f"\nERROR: {error}")
    except (ValueError, KeyError) as error:
        print(f"\nERROR: {error}")
    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
