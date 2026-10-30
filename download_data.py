from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    movies = DATA_DIR / "ml-latest-small" / "movies.csv"
    ratings = DATA_DIR / "ml-latest-small" / "ratings.csv"

    if movies.exists() and ratings.exists():
        print("MovieLens dataset already downloaded and extracted.")
        print(f"Movies: {movies}")
        print(f"Ratings: {ratings}")
        return

    print("Downloading MovieLens latest-small dataset...")
    urlretrieve(URL, ZIP_PATH)
    print("Extracting dataset...")
    with ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Done.")
    print(f"Movies: {movies}")
    print(f"Ratings: {ratings}")
    print("Dataset source: GroupLens MovieLens latest-small.")

if __name__ == "__main__":
    main()
