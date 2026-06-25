# MiniLM Semantic Search

A simple semantic search project built using Sentence Transformers and cosine similarity.

Instead of matching exact keywords, the system converts documents and user queries into embeddings and retrieves the most semantically similar documents.

## Features

* MiniLM sentence embeddings (`all-MiniLM-L6-v2`)
* Cosine similarity based retrieval
* Top-K document ranking
* Similarity threshold filtering
* Interactive command-line search

## How It Works

1. Load NLP-related documents from a CSV file.
2. Convert all documents into embeddings using MiniLM.
3. Convert the user query into an embedding.
4. Compute cosine similarity between the query and all documents.
5. Return the most relevant documents above a similarity threshold.

## Example Output
<img width="736" height="460" alt="image" src="https://github.com/user-attachments/assets/29fedc4c-9365-4882-98e9-f78d976991b7" />

## Technologies Used

* Python
* Sentence Transformers
* MiniLM (`all-MiniLM-L6-v2`)
* Scikit-learn
* Pandas

## Dataset

The project uses a custom CSV dataset containing NLP concepts such as:

* TF-IDF
* Word Embeddings
* BERT
* MiniLM
* RAG
* FAISS
* ChromaDB
* LangChain

## Future Improvements

* Use FAISS for faster retrieval
* Load documents from PDFs
* Add a Streamlit interface
* Increase dataset size for better retrieval quality



