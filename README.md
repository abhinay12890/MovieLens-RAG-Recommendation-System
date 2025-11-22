# MovieLens-RAG-Recommendation-System

## Project Overview
This project focuses on building a movie recommendation engine using the MovieLens dataset. The system combined with:
  - Regex + SpaCy based text preprocessing
  - Sentence Transformer embeddings
  - FAISS vector similarity search
  - Chunk-level Retrieval-Augmented Generation (RAG)
  - llama-3.1-8b-instant for natural-language recommendations
  - Streamlit application for RAG.
The goal is to provide natural-language movie recommendations while restricting outputs to retrived metadata.

---
## Dataset
- **Source:** [MovieLens- ml-latest.zip](https://grouplens.org/datasets/movielens/latest/) contains files `genome-scores.csv`,`genome-tags.csv`,`links.csv`,`movies.csv`,`ratings.csv`,`tags.csv` along with **README** file containing information about the csv files.
- **Size:** 50k+ movie titles (after merging all csv files)
- **Features:**
  -  Movie metadata (title,genres,tags)
  -  Aggregated average rating (per movie)
  -  User-generated tag relevance scores
---
## File Structure 
```
├── movielens_final_embeddings            # directory containing stored faiss index
├── MovieLens_RAG_Recommendation_System.ipynb   # Project code (EDA+ Data Cleaning+ Sample RAG)
├── best_churn_model.pkl                  # Final Trained LightGBM model
├── movie_reco.py                         # RAG based Streamlit application
├── requirements.txt                      # List of dependencies for the project
├── README.md                             # Project documentation (this file)
```
---
## Data Preprocessing & Cleaning
- Merged `movies`,`ratings`, and `tags` by `movieId`.
- Aggregated rating by averaging across users.
- Combined & deduplicated tags for each movie.
- Pre-processsing steps on text:
  1. Regex-Based Tag Cleaning (re)
     - remove URLs
     - fixing spacing
     - removing punctuation
     - preserve apostrophes
     - Remove non-ASCII noise
     - Deduplicate tokens while keeping original order
  2. Spacy based batched-tokenization
- Constructed final documents with
  `title,genres,rating,tags`
---
## Exploratory Data Analysis (EDA)
- Genre Distribution: Barplot of movie counts per genre.
- Ratings Histogram : Distribution of average ratings across movies.
- Word Clouds: Generated separately for positive (≥3 rating) and negative (<3 rating) tags.
---
## Chunking Strategy
- To imporove retrieval quality and avoid context overflow
  * only movie tags are chunked in the following structure
    `chunk_text = (
    f"Title: {row['title']}\n"
    f"Genres: {row['genres']}\n"
    f"Tags: {chunk}")`
  * Storing metda data per chunk
    `{
    "movie_id": i,
    "Title": row["title"],
    "Genres": row["genres"],
    "Rating": row["rating"],
    "chunk_id": chunk_id}`
    This ensures multiple tag chunks -> same movie

## Vector embeddings & Indexing
- Usage of Langchain's HuggingFace Embeddings with `all-MiniLM-L6-v2` model
- Storing vectors through `langchain_community.vectorstores.FAISS` and saved index.
- Implemented query → embedding → vector search → top-K retrieval.
---
## RAG + LLM Recommendation Pipeline
1. User enters a natural-language query.
2. Query converted into an embedding.
3. FAISS index retrieves top-K similar movies.
4. Retrieved movie metadata injected as context into an LLM prompt.
5. LLM outputs recommendations in descending order of avg_rating, strictly using retrieved context. using `llama-3.1-8b-instant (via ChatGroq)`
---
## Output
  * Top 7 unique movies
  * Sorted by rating
  * Summary based only on retrieved context
---
## Deployment
- The complete RAG system is wrapped in Streamlit app:
  * Loads saved FAISS index
  * Loads HuggingFace embedding model
  * Performs retrieval + deduplication
  * Uses Gemini-2.5-Flash Model for LLM Output
---
## Insights
* Chunk-level tag embeddings drastically improve semantic retrieval.
* Deduplication ensures LLM receives clean & unique movie entries.
* FAISS enables scalable searching over 50k+ movies.
* Regex + SpaCy cleaning significantly improves retrieval quality.
* Strict “no hallucination” prompt ensures outputs remain grounded.
---
# Usage
- `git clone this repository`
- run `streamlit run movie_reco.py`
---
## Author
**Abhinay Kalavakuri**

