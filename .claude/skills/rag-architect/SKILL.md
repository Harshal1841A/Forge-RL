---
name: "rag-architect"
description: "Design and implement Retrieval-Augmented Generation (RAG) systems. Use when building document Q&A systems, knowledge bases, semantic search, or any system that retrieves context to augment LLM responses. Covers chunking strategies, embeddings, vector databases, retrieval methods, and evaluation frameworks."
---

# RAG Architect

**Tier:** POWERFUL  
**Category:** Engineering  
**Domain:** AI / Information Retrieval / LLM Systems

## Overview

Comprehensive guide for designing and implementing retrieval-augmented generation systems. Covers the entire RAG ecosystem from document chunking strategies to evaluation frameworks, including embedding model selection, vector database comparison, retrieval methods, query enhancement techniques, and production considerations.

## Core Components

### 1. Document Chunking Strategies

Chunking strategy significantly impacts retrieval quality. Choose based on content type:

| Strategy | Best For | Tradeoffs |
|----------|----------|-----------|
| **Fixed-size** | Uniform documents, simple implementation | Fast, predictable; may split semantic units |
| **Sentence-based** | Conversational text, Q&A | Preserves meaning; variable chunk size |
| **Paragraph-based** | Articles, documentation | Natural boundaries; may be too large |
| **Semantic** | Mixed content types | Best quality; higher compute cost |
| **Recursive** | Hierarchical documents | Adapts to structure; complex to tune |
| **Document-aware** | PDFs, HTML, markdown | Respects format; format-specific parsers needed |

#### Fixed-Size Chunking
```python
def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap for context continuity
    return chunks
```

#### Semantic Chunking
```python
from sentence_transformers import SentenceTransformer
import numpy as np

def chunk_semantic(text: str, threshold: float = 0.8) -> list[str]:
    """Split on semantic boundaries using embedding similarity."""
    sentences = text.split('. ')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i])
        if similarity < threshold:  # Semantic boundary detected
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append('. '.join(current_chunk))
    return chunks
```

#### Chunk Size Guidelines
- **128-256 tokens**: Fine-grained retrieval, high precision, may lose context
- **512 tokens**: Good balance for most use cases (recommended default)
- **1024 tokens**: Better context, lower precision, slower retrieval
- **2048+ tokens**: Full sections, use for document-level retrieval only

### 2. Embedding Models

Choose embedding dimensions based on use case:

| Dimension | Models | Best For |
|-----------|--------|----------|
| **128-256** | all-MiniLM-L6-v2 | Fast retrieval, lower memory, simple tasks |
| **384-512** | all-mpnet-base-v2, text-embedding-3-small | General-purpose, good balance |
| **768-1024** | text-embedding-ada-002, E5-large | High-quality semantic search |
| **2048+** | Specialized domain models | Domain-specific (medical, legal, code) |

#### Model Selection
```python
# OpenAI (recommended for production)
from openai import OpenAI
client = OpenAI()

def embed_openai(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # 1536 dims, cost-effective
        # model="text-embedding-3-large"  # 3072 dims, higher quality
    )
    return response.data[0].embedding

# Open-source (for self-hosted / cost control)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # Strong open-source option
embeddings = model.encode(texts, normalize_embeddings=True)
```

### 3. Vector Databases

| Database | Best For | Key Feature |
|----------|----------|-------------|
| **Pinecone** | Managed, production scale | Serverless, managed infra, simple API |
| **Weaviate** | Open-source, rich features | GraphQL API, multi-modal, hybrid search |
| **Qdrant** | High performance, filtering | Rust-based, fast filtered search |
| **Chroma** | Development, prototyping | Lightweight, embedded, easy setup |
| **pgvector** | Existing PostgreSQL users | SQL integration, familiar ops |
| **Milvus** | Enterprise, large scale | Billion-scale, cloud-native |

#### Chroma (Development)
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "doc.pdf", "page": i} for i in range(len(chunks))]
)

# Query
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"source": "doc.pdf"}  # Metadata filtering
)
```

#### Pinecone (Production)
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")
index = pc.create_index(
    name="knowledge-base",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Upsert with metadata
index.upsert(vectors=[
    {"id": f"chunk_{i}", "values": embedding, "metadata": {"text": chunk, "source": "doc.pdf"}}
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
])

# Query with filtering
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"source": {"$eq": "doc.pdf"}},
    include_metadata=True
)
```

### 4. Retrieval Methods

#### Dense Retrieval (Semantic Search)
```python
def dense_retrieval(query: str, k: int = 5) -> list[dict]:
    query_embedding = embed(query)
    results = vector_db.query(vector=query_embedding, top_k=k)
    return results
```

#### Sparse Retrieval (BM25 Keyword Search)
```python
from rank_bm25 import BM25Okapi

def sparse_retrieval(query: str, corpus: list[str], k: int = 5) -> list[str]:
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    top_k_indices = scores.argsort()[-k:][::-1]
    return [corpus[i] for i in top_k_indices]
```

#### Hybrid Retrieval (Recommended)
```python
def hybrid_retrieval(query: str, k: int = 5, alpha: float = 0.7) -> list[dict]:
    """Combine dense and sparse retrieval with weighted fusion."""
    dense_results = dense_retrieval(query, k=k*2)
    sparse_results = sparse_retrieval(query, k=k*2)

    # Reciprocal Rank Fusion
    scores = {}
    for rank, result in enumerate(dense_results):
        scores[result['id']] = scores.get(result['id'], 0) + alpha / (rank + 60)
    for rank, result in enumerate(sparse_results):
        scores[result['id']] = scores.get(result['id'], 0) + (1-alpha) / (rank + 60)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids[:k]
```

#### Reranking (Cross-Encoder)
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
    """Rerank retrieved candidates using a cross-encoder."""
    pairs = [[query, candidate] for candidate in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [text for text, score in ranked[:top_k]]
```

### 5. Query Enhancement Techniques

#### HyDE (Hypothetical Document Embeddings)
```python
def hyde_retrieval(query: str) -> list[dict]:
    """Generate a hypothetical answer and embed it for retrieval."""
    hypothetical = llm.complete(
        f"Write a detailed answer to: {query}\n\nAnswer:"
    )
    hyde_embedding = embed(hypothetical)
    return vector_db.query(vector=hyde_embedding, top_k=5)
```

#### Multi-Query Generation
```python
def multi_query_retrieval(query: str) -> list[dict]:
    """Generate multiple query perspectives for better recall."""
    variations = llm.complete(f"""Generate 3 different search queries for:
Original: {query}
Variations (one per line):""").split('\n')

    all_results = []
    for q in [query] + variations:
        all_results.extend(dense_retrieval(q, k=3))

    # Deduplicate by ID
    seen = set()
    unique_results = []
    for r in all_results:
        if r['id'] not in seen:
            seen.add(r['id'])
            unique_results.append(r)

    return unique_results[:5]
```

#### Step-Back Prompting
```python
def step_back_retrieval(query: str) -> list[dict]:
    """Abstract query to higher concept for broader context retrieval."""
    abstract_query = llm.complete(
        f"What is the broader concept or principle behind: {query}\n\nAbstract concept:"
    )
    # Retrieve on both specific and abstract queries
    specific_results = dense_retrieval(query, k=3)
    abstract_results = dense_retrieval(abstract_query, k=2)
    return specific_results + abstract_results
```

### 6. RAG Pipeline Assembly

```python
class RAGPipeline:
    def __init__(self, vector_db, llm, reranker=None):
        self.vector_db = vector_db
        self.llm = llm
        self.reranker = reranker

    def query(self, question: str, k: int = 5) -> str:
        # 1. Retrieve
        results = hybrid_retrieval(question, k=k*2)

        # 2. Rerank (optional)
        if self.reranker:
            contexts = [r['text'] for r in results]
            contexts = rerank(question, contexts, top_k=k)
        else:
            contexts = [r['text'] for r in results[:k]]

        # 3. Generate
        context_str = "\n\n---\n\n".join(contexts)
        prompt = f"""Answer the question based on the provided context.
If the context doesn't contain the answer, say "I don't have information about this."

Context:
{context_str}

Question: {question}

Answer:"""

        return self.llm.complete(prompt)
```

## Quality Assurance and Evaluation

### Evaluation Metrics

```python
# Using RAGAS framework
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# Key metrics:
# Faithfulness: Is the answer grounded in the retrieved context?
# Answer Relevancy: Does the answer address the question?
# Context Precision: Is the retrieved context relevant?
# Context Recall: Is all relevant information retrieved?
```

### Evaluation Thresholds

| Metric | Target | Action if Below |
|--------|--------|-----------------|
| Faithfulness | > 0.85 | Tighten prompt, improve chunking |
| Answer Relevancy | > 0.80 | Improve retrieval, check embeddings |
| Context Precision | > 0.75 | Tune retrieval k, add reranker |
| Context Recall | > 0.70 | Increase k, improve chunking |

## Production Considerations

### Caching

```python
import hashlib
from functools import lru_cache

# Cache embeddings
@lru_cache(maxsize=10000)
def cached_embed(text: str) -> tuple:
    return tuple(embed(text))

# Cache retrieval results (Redis)
import redis
r = redis.Redis()

def cached_retrieval(query: str, ttl: int = 3600) -> list[dict]:
    key = f"rag:{hashlib.md5(query.encode()).hexdigest()}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    results = hybrid_retrieval(query)
    r.setex(key, ttl, json.dumps(results))
    return results
```

### Streaming Responses

```python
async def stream_rag_response(question: str):
    contexts = await retrieve_async(question)
    context_str = format_contexts(contexts)
    prompt = build_prompt(question, context_str)

    async for chunk in llm.stream(prompt):
        yield chunk
```

### Fallback Mechanisms

```python
def rag_with_fallback(question: str) -> str:
    try:
        results = retrieve(question, k=5)
        if not results or max(r['score'] for r in results) < 0.5:
            # Low confidence retrieval — use LLM knowledge directly
            return llm.complete(f"Answer based on your knowledge: {question}")
        return generate_answer(question, results)
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        return "I'm unable to answer this question right now."
```

### Cost Optimization

```python
# Use cheaper embeddings for retrieval, expensive for reranking
retrieval_model = "text-embedding-3-small"   # $0.02/1M tokens
reranking_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Local, free

# Batch embedding requests
def batch_embed(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=retrieval_model)
        embeddings.extend([e.embedding for e in response.data])
    return embeddings
```

### Safety and Guardrails

```python
def safe_rag_query(question: str) -> str:
    # Content filtering
    if contains_harmful_content(question):
        return "I cannot answer this question."

    results = retrieve(question)

    # Hallucination detection
    answer = generate_answer(question, results)
    if not is_grounded_in_context(answer, results):
        return "I don't have reliable information about this."

    # PII filtering
    return redact_pii(answer)
```

## Architecture Patterns

### Naive RAG (Starting Point)
```
Query → Embed → Vector Search → Top-K Chunks → LLM → Response
```

### Advanced RAG (Production)
```
Query → Query Enhancement → Hybrid Retrieval → Reranking → 
Context Compression → LLM with Citations → Hallucination Check → Response
```

### Agentic RAG (Complex Tasks)
```
Query → Planning Agent → [Iterative Retrieval + Reasoning] → 
Synthesis Agent → Verification Agent → Response
```

## Common Pitfalls

1. **Chunk size too large** — reduces retrieval precision; start at 512 tokens
2. **No overlap between chunks** — loses context at boundaries; use 10-20% overlap
3. **Ignoring metadata filtering** — slows retrieval; always filter by document source/date
4. **Not evaluating retrieval separately** — measure retrieval quality independently from generation
5. **Single query only** — use multi-query or HyDE for better recall
6. **No reranking** — initial retrieval is imprecise; always rerank top candidates
7. **Static embeddings for dynamic content** — re-embed when documents change
8. **Missing citation tracking** — always track which chunks generated which parts of the answer
