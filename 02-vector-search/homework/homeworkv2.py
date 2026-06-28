#!/usr/bin/env python3
"""
LLM Zoomcamp - Homework 2: Vector Search
Complete solution script to run locally.

Prerequisites:
    pip install onnxruntime tokenizers numpy tqdm minsearch gitsource huggingface-hub

Also download embedder.py and download.py from:
    https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed/

Then run: python download.py  (to fetch the ONNX model)
"""

import numpy as np
from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
import minsearch

# =============================================================================
# SETUP
# =============================================================================

print("Loading embedder...")
embedder = Embedder()

print("Fetching documents from GitHub...")
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
print(f"Loaded {len(documents)} documents")

# =============================================================================
# Q1. EMBEDDING A QUERY
# =============================================================================

query_q1 = "How does approximate nearest neighbor search work?"
v_q1 = embedder.encode(query_q1)
print(f"\n=== Q1 ===")
print(f"Query: {query_q1}")
print(f"Vector shape: {v_q1.shape}")
print(f"First value v[0]: {v_q1[0]:.4f}")

# =============================================================================
# Q2. COSINE SIMILARITY
# =============================================================================

# Find the specific document
target_file = "02-vector-search/lessons/07-sqlitesearch-vector.md"
target_doc = None
for doc in documents:
    if doc["filename"] == target_file:
        target_doc = doc
        break

v_doc = embedder.encode(target_doc["content"])
# Since vectors are normalized, dot product = cosine similarity
cosine_sim = float(np.dot(v_doc, v_q1))
print(f"\n=== Q2 ===")
print(f"File: {target_file}")
print(f"Cosine similarity with Q1 query: {cosine_sim:.4f}")

# =============================================================================
# Q3. CHUNKING AND SEARCH BY HAND
# =============================================================================

print(f"\n=== Q3 ===")
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Total chunks: {len(chunks)}")

# Embed all chunks in batches
print("Embedding all chunks (this may take a moment)...")
chunk_texts = [chunk["content"] for chunk in chunks]
chunk_vectors = embedder.encode_batch(chunk_texts)
X = np.array(chunk_vectors)  # Matrix of all chunk embeddings

# Score against Q1 query
scores = X.dot(v_q1)
best_idx = int(np.argmax(scores))
best_chunk = chunks[best_idx]
print(f"Highest score: {scores[best_idx]:.4f}")
print(f"Best chunk file: {best_chunk['filename']}")
print(f"Best chunk start: {best_chunk['start']}")

# =============================================================================
# Q4. VECTOR SEARCH WITH MINSEARCH
# =============================================================================


# Build vector index
chunk_vectors_array = np.vstack(chunk_vectors)

vector_index = minsearch.VectorSearch(
    keyword_fields=["filename", "start"],
)
vector_index.fit(chunk_vectors_array, chunks)

query_q4 = "What metric do we use to evaluate a search engine?"
vector_results_q4 = vector_index.search(embedder.encode(query_q4), num_results=5)
print(f"Query: {query_q4}")
print(f"First result file: {vector_results_q4[0]['filename']}")

# =============================================================================
# Q5. TEXT SEARCH VS VECTOR SEARCH
# =============================================================================

print(f"\n=== Q5 ===")

# Build text index
text_index = minsearch.Index(text_fields=["content"])
text_index.fit(chunks)

query_q5 = "How do I store vectors in PostgreSQL?"

vector_results_q5 = vector_index.search(embedder.encode(query_q5), num_results=5)
text_results_q5 = text_index.search(query_q5, num_results=5)

vector_files = {r["filename"] for r in vector_results_q5}
text_files = {r["filename"] for r in text_results_q5}

only_in_vector = vector_files - text_files

print(f"Query: {query_q5}")
print(f"\nVector search top 5 files:")
for i, r in enumerate(vector_results_q5, 1):
    print(f"  {i}. {r['filename']}")

print(f"\nText search top 5 files:")
for i, r in enumerate(text_results_q5, 1):
    print(f"  {i}. {r['filename']}")

print(f"\nFile(s) in vector results but NOT in text results:")
for f in only_in_vector:
    print(f"  -> {f}")

# =============================================================================
# Q6. HYBRID SEARCH (RRF)
# =============================================================================

print(f"\n=== Q6 ===")

def rrf(result_lists, k=60, num_results=5):
    """Reciprocal Rank Fusion"""
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

query_q6 = "How do I give the model access to tools?"
query_vector_q6 = embedder.encode(query_q6)

vector_results_q6 = vector_index.search(query_vector_q6, num_results=10)
text_results_q6 = text_index.search(query_q6, num_results=10)

print(f"Query: {query_q6}")
print(f"\nVector search top 5:")
for i, r in enumerate(vector_results_q6[:5], 1):
    print(f"  {i}. {r['filename']}")

print(f"\nText search top 5:")
for i, r in enumerate(text_results_q6[:5], 1):
    print(f"  {i}. {r['filename']}")

fused_results = rrf([vector_results_q6, text_results_q6], k=60, num_results=5)
print(f"\nRRF fused top 5:")
for i, r in enumerate(fused_results, 1):
    print(f"  {i}. {r['filename']}")

print(f"\n>>> Q6 Answer: First RRF result is '{fused_results[0]['filename']}'")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("SUMMARY OF ANSWERS")
print("="*60)
print(f"Q1. v[0] = {v_q1[0]:.4f}  ->  closest to -0.02")
print(f"Q2. Cosine similarity = {cosine_sim:.4f}  ->  closest to 0.37")
print(f"Q3. Best chunk file = {best_chunk['filename']}")
print(f"Q4. First result file = {vector_results_q4[0]['filename']}")
print(f"Q5. In vector only = {list(only_in_vector)[0] if only_in_vector else 'None'}")
print(f"Q6. First RRF result = {fused_results[0]['filename']}")