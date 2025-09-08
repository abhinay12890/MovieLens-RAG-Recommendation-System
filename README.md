# Retrieval-Augmented-Movie-Recommendation-System-NLP-FAISS-LLM

## Project Overview

This project focuses on building a movie recommendation engine using the MovieLens dataset.
The system combines NLP preprocessing, sentence transformer embeddings, FAISS similarity search, and a Retrieval-Augmented Generation (RAG) pipeline with an LLM.
The goal is to provide natural-language movie recommendations while avoiding hallucinations by restricting outputs to retrieved metadata.
---
## Dataset
- **Source:** [MovieLens- ml-latest.zip](https://grouplens.org/datasets/movielens/latest/) contains files
- `genome-scores.csv`,`genome-tags.csv`,`links.csv`,`movies.csv`,`ratings.csv`,`tags.csv` along with **README** file containing information about the csv files.
- **Size:** 50k+ movie titles (after merging all csv files)
- **Features:**
  -  Movie metadata (title,genres,tags)
  -  Aggregated average rating (per movie)
  -  User-generated tag relevance scores
---
## Data Preprocessing
- Merged `movies`,`ratings`, and `tags` by `movieId`.
- Aggregated rating by averaging across users.
- Combined & deduplicated tags for each movie.
- Pre-processsing steps on text:
  -  Tokenization
  -  POS-tagged lemmatization (WordNet)
  -  Removal of stopwords and punctuation
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

Input Query: "Recommend a sci-fi action movie with time travel."
Retrieved Movies: 
[ID: 123] Sync (2014) | Genres: Action, Sci-Fi | AvgRating: 4.17 | Tags: scifi chase
[ID: 456] Primer (2004) | Genres: Sci-Fi, Thriller | AvgRating: 3.95 | Tags: time-loop mind-bending
