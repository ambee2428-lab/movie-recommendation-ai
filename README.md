# CineMatch AI — Hybrid Movie Recommendation System

A local Python + Streamlit project that combines:
- Content-based filtering with TF-IDF and cosine similarity.
- Collaborative signals from users who rated selected favourite movies highly.
- A hybrid score, genre/rating/year filters, and an interactive dark-themed UI.
- A small offline ranking evaluation script.

## Important dataset note

This starter version uses the official GroupLens MovieLens `ml-latest-small` dataset. It includes approximately 9,700 movies and 100,000 ratings. The metadata includes movie titles and genres, but does **not** provide reliable language labels. It can contain international films, but it does not guarantee broad coverage of all languages. Language filtering and language-specific coverage require a richer metadata dataset.

Dataset page: https://grouplens.org/datasets/movielens/latest/

## Requirements

- Windows 10/11
- Python 3.10–3.12 recommended
- Internet access for the initial dataset download
- About 1 GB free disk space is ample for this small version

## Setup in Windows CMD

Open Command Prompt and move to the folder where you want the project:

```bat
cd %USERPROFILE%\Desktop
mkdir movie_recommendation_ai
cd movie_recommendation_ai
```

Download or copy the project files into this folder, preserving the folder structure shown below.

Create and activate a virtual environment:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, use an installed compatible version, for example `py -3.12 -m venv .venv`.

Download and extract MovieLens:

```bat
python download_data.py
```

Train and save the recommendation model:

```bat
python train.py
```

Run the offline evaluation:

```bat
python evaluate.py
```

Start the Streamlit app:

```bat
python -m streamlit run app.py
```

The app should open in your browser. If it does not, open the local URL printed in the terminal, usually `http://localhost:8501`.

## Project structure

```text
movie_recommendation_ai/
│   app.py
│   download_data.py
│   train.py
│   evaluate.py
│   requirements.txt
│   README.md
│
├── src/
│   ├── __init__.py
│   └── recommender.py
│
├── data/
│   └── raw/
│       └── ml-latest-small/
│           ├── movies.csv
│           ├── ratings.csv
│           ├── links.csv
│           └── tags.csv
│
└── models/
    └── movie_recommender.joblib
```

The `data` and `models` files are created by the download/training commands.

## Libraries

- `pandas`: reads and manipulates CSV data.
- `numpy`: numerical operations.
- `scikit-learn`: TF-IDF and cosine similarity.
- `scipy`: sparse matrix support used by scikit-learn.
- `joblib`: saves and loads the fitted model artifacts.
- `streamlit`: interactive web interface.

## Recommendation method

For a movie candidate \(m\) and selected favourites \(S\):

1. Build a TF-IDF representation from each movie title and its genres.
2. Calculate content similarity between candidate \(m\) and the selected favourites.
3. Find users who gave at least 4/5 to any selected favourite.
4. Aggregate those users' ratings of candidate movies into a collaborative signal.
5. Normalize both signals and combine them:

\[
H(m)=w_c C(m)+(1-w_c)R(m)
\]

Here, \(C(m)\) is normalized content similarity, \(R(m)\) is normalized collaborative score, and \(w_c\) is the user-adjustable content weight. If no usable collaborative signal exists, the app falls back to content-based recommendations.

A small popularity adjustment is applied after the blend. The final score is a ranking score, not a probability.

## Evaluation

Run:

```bat
python evaluate.py
```

The script uses a small, deterministic sample of users, holds out each selected user's latest rating, and reports Precision@10, Recall@10 and Hit Rate@10 for held-out movies rated at least 4/5. These metrics are dependent on this sample and split strategy; they should not be treated as guaranteed real-world performance. For formal research, use a larger temporal split and compare against popularity and content-only baselines.

## Common problems

**`python` is not recognized**
- Reinstall Python from https://www.python.org/downloads/windows/
- During installation, select “Add Python to PATH”, then reopen CMD.
- Try `py --version`.

**`No module named streamlit`**
```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**`Dataset not found`**
Run `python download_data.py` from the project root, not from inside `src`.

**`Model not found`**
Run `python train.py` before starting Streamlit.

**App changes do not show**
Stop Streamlit with `Ctrl+C`, then restart:
```bat
python -m streamlit run app.py
```

## Next improvements

- Use a metadata source with explicit original language and country fields.
- Add poster images via TMDB IDs, subject to TMDB API terms and attribution rules.
- Add user accounts and save ratings/preferences in SQLite.
- Compare content-only, collaborative-only and hybrid ranking metrics.
- For large datasets, use a sampled or incremental training strategy and measure memory usage.
