"""
train.py — data processing and model training pipeline.

    python3 train.py

Reads the raw Book-Crossing CSVs from data/, filters them to active users
and popular books, and writes two artefacts to models/:

  pivot_table.parquet  — (book × user) rating matrix used by the KNN model
  books_data.parquet   — book title → ISBN / author lookup for the UI

The fitted NearestNeighbors model is NOT persisted to disk.  Because it
stores only its training data (the pivot table), re-fitting from parquet at
app load time is fast (~1 s) and avoids sklearn version-mismatch issues.

Run from the project root so that relative paths resolve correctly.
"""

import os

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR    = "data"
MODELS_DIR  = "models"
MIN_USER_RATINGS = 200
MIN_BOOK_RATINGS = 50

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    """Read Books.csv and Ratings.csv from DATA_DIR and return them as DataFrames."""
    print("Loading CSVs …")
    books   = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"),   encoding="latin-1", on_bad_lines="skip")
    ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), encoding="latin-1")
    print(f"  Books   : {books.shape[0]:,} rows")
    print(f"  Ratings : {ratings.shape[0]:,} rows")
    return books, ratings


def build_pivot(books, ratings):
    """
    Filter ratings to active users / popular books and build a (book × user) pivot.

    Filtering thresholds are controlled by MIN_USER_RATINGS and MIN_BOOK_RATINGS.
    Unrated cells are filled with 0, which the cosine-distance metric treats as
    "no signal" rather than a negative rating.

    Returns (pivot DataFrame, merged DataFrame with title/author columns).
    """
    print("Filtering and building pivot table …")

    user_counts  = ratings["User-ID"].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index

    book_counts   = ratings["ISBN"].value_counts()
    popular_isbns = book_counts[book_counts >= MIN_BOOK_RATINGS].index

    filtered = ratings[
        ratings["User-ID"].isin(active_users) &
        ratings["ISBN"].isin(popular_isbns)
    ]

    merged = (
        filtered
        .merge(books[["ISBN", "Book-Title", "Book-Author"]], on="ISBN", how="left")
        .dropna(subset=["Book-Title"])
        .drop_duplicates(["User-ID", "Book-Title"])
    )

    pivot = merged.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating",
        fill_value=0,
    )

    n_books, n_users = pivot.shape
    sparsity = 1 - len(merged) / (n_books * n_users)
    print(f"  Pivot   : {n_books:,} books  x  {n_users:,} users  (sparsity {sparsity:.1%})")
    return pivot, merged


def train_model(pivot):
    """
    Fit a brute-force KNN model on the pivot table as a smoke-test.

    The model itself is not saved — app.py re-fits from the parquet at load
    time.  This call exists to catch data-shape errors before writing files.
    """
    print("Training NearestNeighbors …")
    sparse = csr_matrix(pivot.values)
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(sparse)
    print(f"  Fitted on {sparse.shape[0]:,} books")
    return model


def save_artefacts(pivot, books):
    """
    Persist the pivot table and book metadata to Parquet files in MODELS_DIR.

    Parquet is used instead of pickle because it is language-agnostic,
    compresses well, and has no dependency on a specific library version.
    """
    print("Saving artefacts …")

    # NearestNeighbors stores nothing but its training data, so pickling it
    # creates a hard sklearn-version dependency with no benefit.
    # The app reconstructs the model at load time from the pivot table instead.
    pivot.to_parquet(os.path.join(MODELS_DIR, "pivot_table.parquet"))

    book_metadata = (
        books[["ISBN", "Book-Title", "Book-Author"]]
        .drop_duplicates("Book-Title")
        .set_index("Book-Title")
    )
    book_metadata = book_metadata[book_metadata.index.isin(pivot.index)]
    book_metadata.to_parquet(os.path.join(MODELS_DIR, "books_data.parquet"))

    for fname in ["pivot_table.parquet", "books_data.parquet"]:
        kb = os.path.getsize(os.path.join(MODELS_DIR, fname)) / 1024
        print(f"  {fname:<25}  {kb:>8,.0f} KB")


if __name__ == "__main__":
    books, ratings = load_data()
    pivot, merged  = build_pivot(books, ratings)
    train_model(pivot)          # smoke-test fit; app re-fits from parquet at load time
    save_artefacts(pivot, books)
    print("\nDone. Run:  streamlit run app.py")
