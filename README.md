# MovieLens-RAG-Recommendation-System

## Project Overview
This project implements a large-scale Retrieval-Augmented Generation (RAG) movie recommendation system built on the MovieLens dataset.
The system retrieves semantically relevant movies using dense embeddings and FAISS, then reranks results using an LLM to produce natural-language, rating-aware recommendations.

This version focuses on: 
  - High-quality semantic retrival at scale (~70k movies)
  - Strong preprocessing and data filtering for recommendation quality
  - Two-stage retriever -> LLM reranker architechure
  - Proper offline evaluation of retriever
  - Streamlit-based interactive UI.
---
## Key Components
  - Regex-based text cleaning for noisy user tags
  - Dense semantic embeddings using **Sentence Transformers**
  - **FAISS** for scalable vector similarity search
  - Genre-based ground-truth evaluation
  - Streamlit UI for real-time recommendations
---
## Dataset
- **Source:** [MovieLens- ml-latest.zip](https://grouplens.org/datasets/movielens/latest/) includes files `genome-scores.csv`,`genome-tags.csv`,`links.csv`,`movies.csv`,`ratings.csv`,`tags.csv` along with **README** file containing information about the csv files.
- **Scale:** ~70k+ movie titles, Millions of ratings (after merging all csv files)
- **Features:**
  -  Movie metadata (title,genres,tags)
  -  Aggregated average rating (per movie)
  -  User-generated tag relevance scores
---
## File Structure 
```
├── archive/                                     # directory containing stored faiss index
  ├── movielens_rag_system_v1.ipynb              # v1 of this Project code (EDA+ Data Cleaning+ Sample RAG) 
├── faiss_index/                                 # directory containing stored faiss index of current version
  ├── index.faiss
  ├── index.pkl
├── movielens_rag_v2.ipynb                       # 
├── movie_reco.py                         # RAG based Streamlit application
├── requirements.txt                      # List of dependencies for the project
├── README.md                             # Project documentation (this file)
```
---
## Data Preprocessing & Filtering
- Merged `movies`,`ratings`, and `tags` by `movieId`.
- Aggregated rating by averaging across users.
- Extracted release year from movie titles.
- Pre-processsing steps on text:
  1. Regex-Based Tag Cleaning (re)
     - remove URLs
     - fixing spacing
     - removing punctuation
     - preserve apostrophes
     - Remove non-ASCII noise
     - Deduplicate tokens while keeping original order
  2. Data-level Filtering (Quality Control)
- To ensure better recommendations and reduce noise the following filters are applied
`Average rating > 2.0`, `Release year >= 1920`
These filters:
  * Remove extremely low-quality entries.
  * Avoid very old films with limited metadata
  * Improve both retrieval relevance and UI recommendations
Filtering is applied before embedding and indexing
---
## Chunking Strategy
- To imporove retrieval quality and avoid context overflow
  * only movie tags are chunked in the following structure
    `chunk_text = (
    f"Genres: {row['genres']}\n"
    f"Tags: {chunk}")`
  * Storing metda data per chunk
    `{
    "movie_id": i,
    "title": row["title"],
    "genres": row["genres"],
    "rating": row["rating"],
    "chunk_id": chunk_id}`
    This ensures multiple tag chunks -> same movie

## Vector embeddings & Indexing
- Usage of Langchain's HuggingFace Embeddings with `all-MiniLM-L6-v2` model
- Storing vectors through `langchain_community.vectorstores.FAISS` and saved index.
- Retrieval Flow
  `query → embedding → FAISS similarity search → top-K retrieval.`
---
## RAG + LLM Recommendation Pipeline
### Online Flow
1. User enters a natural-language query.
2. Query is embedded.
3. FAISS retrieves top-K similar movies (K=20).
4. Retrieved movie metadata injected as context into an LLM.
5. LLM reranks titles by average rating and displays **Top-7** titles with summaries.
---
### LLM Configuration
* Model: Gemini-2.5-Flash
* Role: Reranking + Summarization
* Strictly restricted to retrieved context to prevent hallucinations.
---
## Retriever Evaluation (Offline)
Ground truth construction based on genre-based relevance sets.
Ground truth size statistics:
  * Min: ~1,000 movies
  * Mean: ~12,000 movies
  * Max: ~35,000 movies
---
## Metrics Used
* **Recall@K**: genre coverage
* **Precision@K**: genre purity of top-k results
* **MRR**: early relevance ranking quality
---
## Final RAG Evaluation Metrics

| Recall@100  = 0.0139 | Precision@100 = 0.9059 | MRR = 0.9179
| Recall@300  = 0.0411 | Precision@300 = 0.8891 | MRR = 0.9179
| Recall@500  = 0.0669 | Precision@500 = 0.8780 | MRR = 0.9179
Queries evaluated: 44

**Observations**
 - Precision remains high -> strong genre consistency
 - Recall increases smoothly with K -> healthy retrieval
 - High MRR -> relevant movies appear early in rankings.
---
## Deployment
- The complete RAG system is wrapped in Streamlit app:
  * Loads the saved FAISS index
  * Uses the same embedding model as evaluation
  * Applies semantic retrieval
  * Uses LLM-based (Gemini-2.5-Flash) reranking to display clean, rating-aware recommendations
---
## UI Output
  * Top 7 unique movies
  * Sorted by average rating
  * Natural-language summaries and grounded strictly in retrieved metadata
---
## Key Insights
  * Regex-based cleaning significantly improves semantic embeddings
  * Data-level filtering enhances recommendation quality
  * Large ground truths require large-K evaluation
  * High precision + high MRR indicate strong early relevance
---
# Usage
- `git clone <repository-url>`
- `cd MovieLens-RAG-Recommendation-System`
- `pip install -r requirements.txt`
- `streamlit run movie_reco.py`
---
## Author
**Kalavakuri Abhinay**

