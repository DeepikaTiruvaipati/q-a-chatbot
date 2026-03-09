

# README.md

# Member Message QA System

This project implements a question-answering system over a private dataset of member text messages.
Users can ask natural questions (e.g., *“When is Layla planning her trip?”*, *“What restaurants does Amira like?”*) and the system retrieves relevant messages, extracts structured information, and returns a concise answer.

The system combines semantic search, lightweight NER-based extraction, and intent classification.
It is deployed as a FastAPI service on HuggingFace Spaces.

---

## 1. Project Overview

The workflow consists of:

1. Data preprocessing and cleaning
2. Generating embeddings using SentenceTransformer
3. Building a FAISS index
4. Implementing person resolution (NER + fuzzy matching)
5. Semantic retrieval (top-K message search)
6. Extracting dates, locations, restaurants, and numeric values
7. Synthesizing final answers based on intent
8. Deploying the API with FastAPI + Docker on HuggingFace Spaces

All experimentation is documented in the Jupyter notebook (`members_qa_bot.ipynb`), while the API runs through `main.py` and `app.py`.

---

## 2. Repository Structure

```
.
├── README.md
├── Dockerfile
├── requirements.txt
├── app.py
├── main.py
│
├── members_qa_bot.ipynb      # EDA + embeddings + index creation + extraction logic
├── messages_saved.json        # Cleaned messages dataset
├── message_embeddings.npy     # Model embeddings for each message
├── faiss_index.ivf            # FAISS IVF index (not mandatory but included)
├── idx_by_user.json           # Mapping: user → list of message indices
└── eda_summary.json           # Exploratory analysis summary
```

These files are the minimum required to reproduce the pipeline and run the deployed API.

---

## 3. How the System Works

### 3.1 User Identification

The system identifies which user the question refers to using:

* spaCy NER to extract person names
* Fuzzy matching against canonical user names
* Normalization (lowercase, punctuation removal, unicode normalization)

### 3.2 Semantic Retrieval

* Queries are encoded using `all-MiniLM-L6-v2`
* Embeddings are compared using cosine similarity
* Top-K relevant messages are retrieved (user-specific first, otherwise global)

### 3.3 Information Extraction

From retrieved text, the system extracts:

* Dates (spaCy DATE + dateutil parsing)
* Locations (spaCy GPE/LOC)
* Numbers
* Restaurants (ORG/FAC + heuristics)

### 3.4 Answer Construction

A simple intent detector classifies the question into:

* when
* how_many
* where
* restaurant
* general

Based on intent, the system assembles a concise final answer.

---

## 4. Running Locally

### Install dependencies

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Start API

```
python app.py
```

The API will be available at:

```
http://localhost:7860/docs
```

---

## 5. Deployment (HuggingFace Spaces)

The service is deployed on HuggingFace Spaces using:

* FastAPI
* Docker runtime
* SentenceTransformers
* spaCy

The production API is accessible at:

```
https://<your-space>.hf.space/docs
```

The `/ask` endpoint accepts:

```json
{
  "question": "What restaurants does Amira like?"
}
```

and returns a structured JSON answer.

---

## 6. Submission Items

**GitHub Repository:**
Add your repo URL here

**HuggingFace Deployment:**
Add your HF Spaces link here

---

## 7. Notes

* All logic is local and deterministic
* No online model calls are used
* Designed to match the dataset logic exactly as demonstrated in the notebook
---
