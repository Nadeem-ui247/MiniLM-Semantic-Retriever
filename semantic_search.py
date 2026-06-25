from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("nlp_documents_full.csv")
documents = df["text"].tolist()
embeddings = model.encode(documents)

while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    query_embedding = model.encode([query])

    similarity_matrix = cosine_similarity(embeddings, query_embedding)

    scores = similarity_matrix.flatten()

    threshold = 0.35
    max_results = 3

    count = 0

    print("\nTop Results:\n")

    for idx in scores.argsort()[::-1]:
        if scores[idx] < threshold:
            continue

        print(f"Similarity: {scores[idx]:.4f}")
        print(documents[idx])
        print()

        count += 1

        if count == max_results:
            break

    if count == 0:
        print("No relevant documents found.")
