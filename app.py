import streamlit as st
import pandas as pd
from src.recommender import load_model, recommend

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: #0b0d12; color: #f3f4f6; }
[data-testid="stSidebar"] { background: #11151d; }
h1, h2, h3 { color: #f7f7fb; }
.hero {
    padding: 1.35rem 1.6rem; border-radius: 18px;
    background: linear-gradient(120deg, #201438, #14243b);
    border: 1px solid #34314e; margin-bottom: 1.1rem;
}
.movie-card {
    background: #151923; border: 1px solid #282e3a;
    border-radius: 14px; padding: 14px; min-height: 190px;
}
.small-muted { color: #a8b0c0; font-size: 0.9rem; }
.score { color: #c4b5fd; font-weight: 700; }
div.stButton > button {
    background: #7c3aed; color: white; border: 0; border-radius: 9px;
}
div.stButton > button:hover { background: #6d28d9; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info("Run `python download_data.py` and then `python train.py` in your project folder.")
    st.stop()

movies = model["movies"].copy()
ratings = model["ratings"]
movie_mean = model["movie_mean"]
movie_count = model["movie_count"]

st.markdown("""
<div class="hero">
  <h1>🎬 CineMatch AI</h1>
  <p>Discover your next favourite movie with a hybrid recommendation engine.</p>
  <p class="small-muted">Content similarity + community rating patterns</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Movies", f"{len(movies):,}")
c2.metric("Ratings", f"{len(ratings):,}")
c3.metric("Users", f"{ratings['userId'].nunique():,}")

st.caption(
    "Dataset note: MovieLens latest-small contains movie titles, genres and ratings, "
    "but does not provide reliable language metadata. Language coverage is therefore "
    "not guaranteed, and language filtering is not available in this version."
)

with st.sidebar:
    st.header("Your preferences")
    st.write("Choose one or more movies you enjoy.")
    title_to_id = movies.sort_values("title").set_index("title")["movieId"].to_dict()
    titles = list(title_to_id.keys())
    selected_titles = st.multiselect(
        "Favourite movies",
        options=titles,
        default=[t for t in titles if t.startswith("Toy Story (1995)")][:1],
        max_selections=8,
        placeholder="Search for a movie..."
    )

    all_genres = sorted({
        g for value in movies["genres"].fillna("")
        for g in value.split("|") if g and g != "(no genres listed)"
    })
    genre_filter = st.selectbox("Genre filter", ["All"] + all_genres)
    min_rating = st.slider("Minimum community rating", 0.0, 5.0, 3.0, 0.5)

    known_years = movies["year"].dropna()
    if not known_years.empty:
        min_data_year = int(known_years.min())
        max_data_year = int(known_years.max())
    else:
        min_data_year, max_data_year = 1900, 2026

    year_range = st.slider(
        "Release year range",
        min_value=min_data_year,
        max_value=max_data_year,
        value=(max(min_data_year, 1980), max_data_year)
        if max_data_year >= 1980 else (min_data_year, max_data_year)
    )
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    content_weight = st.slider(
        "Content similarity weight", 0.0, 1.0, 0.55, 0.05,
        help="Higher values favour similar genres and titles; lower values favour community rating patterns."
    )
    run = st.button("✨ Recommend movies", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Recommendations", "Explore movies", "About the AI"])

with tab1:
    if run:
        if not selected_titles:
            st.warning("Select at least one favourite movie in the sidebar.")
        else:
            seed_ids = [int(title_to_id[t]) for t in selected_titles]
            try:
                results = recommend(
                    model,
                    seed_ids,
                    top_n=top_n,
                    genre_filter=genre_filter,
                    min_rating=min_rating,
                    year_min=year_range[0],
                    year_max=year_range[1],
                    content_weight=content_weight
                )
                if results.empty:
                    st.warning("No movies match these filters. Try lowering the minimum rating or widening the year range.")
                else:
                    st.subheader("Picked for you")
                    st.caption("Recommendations are ranked by a combined score, not a guaranteed probability that you will like a movie.")
                    for start in range(0, len(results), 2):
                        cols = st.columns(2)
                        for col, (_, movie) in zip(cols, results.iloc[start:start+2].iterrows()):
                            year = str(int(movie["year"])) if pd.notna(movie["year"]) else "Year unknown"
                            avg = float(movie["average_rating"])
                            with col:
                                st.markdown(f"""
                                <div class="movie-card">
                                  <h3>{movie['title']}</h3>
                                  <p class="small-muted">{movie['genres']}</p>
                                  <p>📅 {year} &nbsp; ⭐ {avg:.1f}/5</p>
                                  <p class="small-muted">{int(movie['rating_count'])} community ratings</p>
                                  <p class="score">Hybrid score: {float(movie['recommendation_score']):.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("Why this recommendation?"):
                                    st.write(f"Content score: {float(movie['content_score']):.3f}")
                                    st.write(f"Collaborative score: {float(movie['collaborative_score']):.3f}")
                                    st.write("Content score measures metadata similarity to your selected movies. Collaborative score reflects ratings from users who rated your selected movies highly.")
            except Exception as exc:
                st.error(f"Recommendation failed: {exc}")
    else:
        st.info("Choose favourite movies and click **Recommend movies** to get your list.")

with tab2:
    st.subheader("Explore the dataset")
    query = st.text_input("Search movie titles", placeholder="e.g. Spider-Man")
    explore = movies.copy()
    if query:
        explore = explore[explore["title"].str.contains(query, case=False, na=False)]
    explore["average_rating"] = explore["movieId"].map(movie_mean)
    explore["rating_count"] = explore["movieId"].map(movie_count).fillna(0).astype(int)
    explore["average_rating"] = explore["average_rating"].fillna(model["global_mean"])
    explore = explore.sort_values(["rating_count", "average_rating"], ascending=False)
    st.dataframe(
        explore[["title", "genres", "year", "average_rating", "rating_count"]].head(200),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("How CineMatch AI works")
    st.markdown("""
    **1. Content-based filtering**  
    Movie titles and genres are transformed into TF-IDF vectors. Cosine similarity measures how closely each movie matches your selected favourites.

    **2. Collaborative filtering**  
    The system finds MovieLens users who rated your selected favourites highly, then aggregates the ratings they gave to other movies.

    **3. Hybrid ranking**  
    Content and collaborative scores are normalized and blended. A small popularity adjustment helps distinguish otherwise similar candidates.

    **4. Limitations**  
    MovieLens ratings are not a representative sample of every audience or country. Its latest-small metadata does not reliably identify movie language, and recommendations are not real-time streaming availability data.
    """)
    st.write(f"Model created: {model.get('trained_at', 'unknown')}")
    st.write("For a stronger evaluation, run `python evaluate.py` and report its Precision@10, Recall@10 and Hit Rate@10.")
