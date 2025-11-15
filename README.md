# MovieLens-RAG-Recommendation-System

## Project Overview
This project focuses on building a movie recommendation engine using the MovieLens dataset. The system combines NLP preprocessing, sentence transformer embeddings, FAISS similarity search and Retrieval-Augmented Generation (RAG) pipeline with an LLM.
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
## Feature Engineering & Indexing
- Encoded final documents into embeddings using Sentence Transformers.
- Built a FAISS index (cosine similarity with normalized vectors).
- Stored metadata (movieId, title, genres, tags, avg_rating) linked to embeddings.
- Implemented query → embedding → vector search → top-K retrieval.
---
## RAG + LLM Recommendation Pipeline
1. User enters a natural-language query.
2. Query converted into an embedding.
3. FAISS index retrieves top-K similar movies.
4. Retrieved movie metadata injected as context into an LLM prompt.
5. LLM outputs recommendations in descending order of avg_rating, strictly using retrieved context.
---
## Model Building
- Embedding Model: Sentence Transformers (all-MiniLM-L6-v2).
- Vector Index: IndexFlatL2
- LLM: `llama-3.1-8b-instant` used in RAG pipeline.
- Self-Evaluating LLM Recommendation system by comparing user query to retrived chunks to judge relevance and rate its own answer (1-5).
---
## Insights
* Successfully integrated semantic search + ratings into ranking.
* FAISS indexing scales to 50k+ movies efficiently.
* Preprocessing + deduplication reduced noise in tag data, improving retrieval quality.
---
## Author
**Abhinay Kalavakuri**

