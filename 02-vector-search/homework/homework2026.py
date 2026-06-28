import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents
from embedder import Embedder
from minsearch import Index, VectorSearch

# ==========================================
# 0. SETUP: Load Data & Initialize Embedder
# ==========================================
print("Loading documents from GitHub...")
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print(f"Loaded {len(documents)} documents.")

print("Initializing Embedder...")
embedder = Embedder()


# ==========================================
# Q1. Embedding a query
# ==========================================
print("\n--- Q1. Embedding a query ---")
query_q1 = "How does approximate nearest neighbor search work?"
v_q1 = embedder.encode(query_q1)

# Format to 2 decimal places to match the options
print(f"First value v[0]: {v_q1[0]:.2f}")


# ==========================================
# Q2. Cosine similarity
# ==========================================
print("\n--- Q2. Cosine similarity ---")
target_filename = "02-vector-search/lessons/07-sqlitesearch-vector.md"
target_doc = next(doc for doc in documents if doc["filename"] == target_filename)

v_doc_q2 = embedder.encode(target_doc["content"])

# Because the embedder returns normalized vectors, dot product equals cosine similarity
similarity_q2 = np.dot(v_q1, v_doc_q2)
print(f"Cosine similarity: {similarity_q2:.2f}")


# ==========================================
# Q3. Chunking and search by hand
# ==========================================
print("\n--- Q3. Chunking and search by hand ---")
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Created {len(chunks)} chunks.")

# Embed all chunks
chunk_texts = [chunk["content"] for chunk in chunks]
chunk_vectors = embedder.encode_batch(chunk_texts)

# Stack into a matrix X
X = np.array(chunk_vectors)

# Score the Q1 query against all chunks
scores_q3 = X.dot(v_q1)
max_idx = np.argmax(scores_q3)

print(f"Highest scoring chunk belongs to: {chunks[max_idx]['filename']}")


# ==========================================
# Q4. Vector search with minsearch
# ==========================================
print("\n--- Q4. Vector search with minsearch ---")
# Prepare documents for minsearch VectorSearch by attaching vectors
minsearch_docs = []
for i, chunk in enumerate(chunks):
    doc = chunk.copy()
    doc["embedding"] = X[i].tolist()
    minsearch_docs.append(doc)

vector_index = VectorSearch(
    documents=minsearch_docs,
    text_fields=["content", "filename"], 
    vector_field="embedding"
)

query_q4 = "What metric do we use to evaluate a search engine?"
v_q4 = embedder.encode(query_q4)

vector_results_q4 = vector_index.search(v_q4, num_results=1)
print(f"First result filename: {vector_results_q4[0]['filename']}")


# ==========================================
# Q5. Text search vs vector search
# ==========================================
print("\n--- Q5. Text search vs vector search ---")
# Setup Text Index
text_index = Index(
    text_fields=["content"],
    keyword_fields=["filename", "start"]
)
text_index.fit(chunks)

query_q5 = "How do I store vectors in PostgreSQL?"
v_q5 = embedder.encode(query_q5)

# Get top 5 results for both
text_results_q5 = text_index.search(query_q5, num_results=5)
vector_results_q5 = vector_index.search(v_q5, num_results=5)

text_filenames = {r["filename"] for r in text_results_q5}
vector_filenames = {r["filename"] for r in vector_results_q5}

# Find file in vector but NOT in text
diff = vector_filenames - text_filenames
print(f"Files in vector results but not text results: {diff}")


# ==========================================
# Q6. Hybrid search
# ==========================================
print("\n--- Q6. Hybrid search ---")
# RRF function provided in the homework
def rrf(result_lists, k=60, num_results=5):
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
v_q6 = embedder.encode(query_q6)

text_results_q6 = text_index.search(query_q6, num_results=5)
vector_results_q6 = vector_index.search(v_q6, num_results=5)

hybrid_results_q6 = rrf([vector_results_q6, text_results_q6])
print(f"First result after RRF: {hybrid_results_q6[0]['filename']}")