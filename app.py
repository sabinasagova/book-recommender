"""
Book Recommendation System — Streamlit frontend.

Loads the KNN model and pivot table produced by notebook.ipynb and
serves recommendations through a simple web UI.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="centered",
)

MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# Artifact loading (cached so the heavy deserialization happens only once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recommendation model…")
def load_artifacts():
    """Load DataFrames from parquet and reconstruct the KNN model in-process."""
    paths = {
        "pivot":      os.path.join(MODELS_DIR, "pivot_table.parquet"),
        "books_data": os.path.join(MODELS_DIR, "books_data.parquet"),
    }

    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing artefacts: {missing}. "
            "Run `python3 train.py` first to build the model."
        )

    pivot_table = pd.read_parquet(paths["pivot"])
    books_data  = pd.read_parquet(paths["books_data"])

    # Reconstruct the model here rather than unpickling it.
    # NearestNeighbors stores only its training data, so re-fitting is cheap
    # (~1 s) and completely avoids sklearn version mismatch warnings.
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(csr_matrix(pivot_table.values))

    return model, pivot_table, books_data


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def get_recommendations(book_name, model, pivot_table, n=5):
    """
    Query the KNN model for books similar to *book_name*.

    Returns
    -------
    (list[dict], None)  on success — each dict has 'title' and 'similarity'.
    (None, str)         on failure — the string is a human-readable error.
    """
    idx = np.where(pivot_table.index == book_name)[0]
    if len(idx) == 0:
        return None, f"'{book_name}' was not found in the model index."

    distances, indices = model.kneighbors(
        pivot_table.iloc[idx[0], :].values.reshape(1, -1),
        n_neighbors=n + 1,  # +1 because the query book is always the first result
    )

    results = [
        {
            "title":      pivot_table.index[indices[0][i]],
            "similarity": round(1.0 - float(distances[0][i]), 4),
        }
        for i in range(1, len(indices[0]))
    ]
    return results, None


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
try:
    model, pivot_table, books_data = load_artifacts()
    book_list = sorted(pivot_table.index.tolist())
    load_error = None
except FileNotFoundError as exc:
    load_error = str(exc)
    book_list = []
    model = pivot_table = books_data = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app applies **Collaborative Filtering** with **K-Nearest Neighbours** "
        "and *cosine similarity* to surface books that share similar rating patterns "
        "across users.\n\n"
        "**Dataset:** Book-Crossing (Kaggle)  \n"
        "**Algorithm:** `sklearn.neighbors.NearestNeighbors`"
    )
    if book_list:
        st.success(f"Model loaded — {len(book_list):,} books available")
    else:
        st.warning("Model not loaded yet")

    st.divider()
    st.caption("Run `python train.py` to (re)generate the model artefacts.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📚 Book Recommendation System")
st.markdown(
    "Select a book from the dropdown (you can type to search) and click "
    "**Show Recommendations** to discover similar reads."
)
st.divider()

if load_error:
    st.error(f"⚠️ {load_error}")
    st.stop()

# Book selector
selected_book = st.selectbox(
    "🔍 Select or type a book title:",
    options=book_list,
    index=0,
    help="Start typing to narrow down the list.",
)

n_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Recommend button
if st.button("Show Recommendations", type="primary", use_container_width=True):
    with st.spinner("Finding similar books…"):
        results, error = get_recommendations(selected_book, model, pivot_table, n=n_recs)

    if error:
        st.error(f"❌ {error}")
    else:
        # Build a title → author lookup from the saved metadata
        title_to_author = books_data["Book-Author"].to_dict()

        st.subheader(f"Books similar to *{selected_book}*")
        st.caption(
            "Ranked by cosine similarity of user-rating vectors — "
            "1.00 = identical rating pattern, 0.00 = no overlap."
        )
        st.divider()

        for rank, rec in enumerate(results, start=1):
            author = title_to_author.get(rec["title"], "Unknown Author")
            col_info, col_score = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{rank}. {rec['title']}**")
                st.caption(f"✍️ {author}")
            with col_score:
                st.metric(label="Similarity", value=f"{rec['similarity']:.2%}")
            st.divider()
