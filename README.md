# Book Recommendation Engine

An item-based collaborative filtering system that recommends books based on latent patterns in user ratings. Built with the [Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) and served through an interactive Streamlit web app.

---

## Project Overview

The engine learns which books share similar audiences by analysing how thousands of readers have rated them. When you select a title, it finds the *k* most similar books in the rating-vector space and returns them ranked by cosine similarity — so a score of `1.00` means an almost identical readership profile, while `0.00` means no overlap at all.

The model catalogue covers roughly **1,964 books**, chosen to balance sparsity against variety (see [The Approach](#the-approach) below).

---

## Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the Book-Crossing dataset from Kaggle and place the three CSV files (`Books.csv`, `Ratings.csv`, `Users.csv`) into a `data/` folder at the project root:

```
book-recommender/
└── data/
    ├── Books.csv
    ├── Ratings.csv
    └── Users.csv
```

### 3. Train the model

Running this script filters the raw data, builds the pivot table, and saves the required `.parquet` artifacts to `models/`:

```bash
python train.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## The Approach

### Algorithm: Item-Based KNN with Cosine Similarity

The core idea is **item-based collaborative filtering**: instead of predicting what a specific user will like, we measure how similar *books* are to one another by comparing how all users have rated them.

1. **Pivot table** — ratings are restructured into a `(book × user)` matrix where each cell holds a user's explicit 1–10 rating (or 0 if they have not rated the book).
2. **Sparse matrix** — the pivot table is converted to a `scipy.sparse.csr_matrix` to keep memory usage manageable.
3. **KNN with cosine distance** — `sklearn.neighbors.NearestNeighbors` is fitted on the sparse matrix. Cosine similarity is used because it measures the *angle* between rating vectors, making it robust to differences in rating scale across users.

### Why ~1,964 books?

The raw dataset contains over 270,000 books but is extremely sparse — most books have very few ratings. Training on the full catalogue would produce a matrix that is >99.9 % zeros, causing the model to surface spurious neighbours.

To improve signal quality, the data is filtered to:

| Filter | Threshold |
|---|---|
| Minimum ratings per user | ≥ 200 |
| Minimum ratings per book | ≥ 50 |

This leaves roughly **1,964 books** rated by roughly **888 active users** — dense enough for meaningful similarity scores, while still covering a broad range of genres and authors.

---

## Future Improvements

| Area | Idea |
|---|---|
| **Implicit zeros** | Unrated entries are currently treated as `0`, which conflates "haven't read" with "rated neutrally". A dedicated implicit-feedback model (e.g. `implicit` library with BPR or ALS) would handle this more correctly. |
| **User mean-centering** | Subtracting each user's mean rating before building the pivot would normalise for individual rating-scale bias (generous vs. harsh raters). |
| **Matrix Factorisation** | SVD or NMF decompose the rating matrix into latent factors, producing denser representations and better cold-start handling than neighbourhood methods. |
| **Larger catalogue** | Lowering the minimum-ratings threshold (at the cost of more sparsity) or applying dimensionality reduction first would expose a wider book selection. |
| **Hybrid approach** | Combining collaborative filtering with content-based signals (genre, author, description embeddings) would help surface relevant books with fewer ratings. |

---

## Tech Stack

| Layer | Library |
|---|---|
| Data processing | `pandas`, `numpy` |
| Sparse matrices | `scipy` |
| KNN model | `scikit-learn` |
| Serialisation | `pyarrow` / Parquet |
| Web UI | `streamlit` |

---

## Repository Structure

```
book-recommender/
├── app.py              # Streamlit web application
├── train.py            # Data processing & model training script
├── notebook.ipynb      # Exploratory data analysis
├── requirements.txt    # Python dependencies
├── models/             # Generated artifacts (git-ignored; created by train.py)
│   ├── pivot_table.parquet
│   └── books_data.parquet
└── data/               # Raw Book-Crossing CSVs (git-ignored; download separately)
```
