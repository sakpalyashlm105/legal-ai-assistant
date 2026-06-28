"""
retrieval/vector_store.py
--------------------------
FAISS vector index with OpenAI embeddings, MMR search, and LLM re-ranking.

What this file does:
    1. Embeds DocumentChunk text using OpenAI text-embedding-3-small.
    2. Stores those embeddings in a FAISS flat L2 index on disk.
    3. At query time, embeds the query and returns the top-K most similar
       chunks using MMR (Maximal Marginal Relevance) to balance relevance
       with diversity.
    4. Optionally re-ranks the top-10 FAISS results to top-3 using
       GPT-4o-mini, unless the top-1 cosine similarity is already > 0.92
       (a near-perfect match that doesn't need re-ranking).

Key concepts:
    FAISS (Facebook AI Similarity Search)
        An in-memory/on-disk index that can search billions of vectors in
        milliseconds. We use IndexFlatIP (inner product on normalized vectors)
        which gives cosine similarity. Think of it as a very fast "nearest
        neighbour" lookup in a 1536-dimensional space.

    MMR (Maximal Marginal Relevance)
        A retrieval strategy that picks results which are relevant to the
        query but diverse from each other. Without MMR, if clause 3 and
        clause 4 are nearly identical and both match the query, you'd get
        both -- MMR would pick only one and use the second slot for a less
        similar but more distinct result, giving you broader coverage.

    LLM re-ranking
        FAISS finds the top-10 candidates by vector similarity. GPT-4o-mini
        then reads those 10 chunks and the original query and re-orders them
        by actual relevance -- catching cases where semantically similar text
        is not actually the right answer to the question. We skip this step
        when the top-1 similarity is > 0.92 (a clear winner -- no need to pay
        for an extra LLM call).

Architecture constraints from CLAUDE.md (locked, do not change):
    - Model: text-embedding-3-small, 1536 dimensions
    - Retrieval: top-10 FAISS -> LLM rerank -> top-3 returned
    - Skip rerank when top-1 cosine similarity > 0.92
    - Batch all clause queries into ONE embedding call (not 10 separate calls)
    - Cache per-document session: don't re-embed the same document twice
      in the same process run

Dependencies:
    faiss-cpu, numpy, openai, tiktoken, schemas/chunk.py
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

from schemas.chunk import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# How many candidates FAISS returns before MMR/re-ranking winnows them
FAISS_TOP_K = 10

# Final number of chunks returned to the caller after re-ranking
RERANK_TOP_N = 3

# If the best FAISS match has cosine similarity above this value, the match
# is clear enough that we skip the LLM re-rank call entirely.
RERANK_SKIP_THRESHOLD = 0.92

# MMR diversity weight. Lambda=1.0 means pure relevance (no diversity penalty);
# Lambda=0.0 means maximum diversity (ignores query relevance entirely).
# 0.7 gives a good balance: prioritise relevance but avoid near-duplicate results.
MMR_LAMBDA = 0.7


# ---------------------------------------------------------------------------
# OpenAI client (lazy init, same pattern as ocr.py)
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Ensure config.py or load_dotenv() "
                "is called before using the vector store."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Per-session embedding cache
# ---------------------------------------------------------------------------
# Maps document_hash -> np.ndarray of shape (num_chunks, EMBEDDING_DIM).
# Cleared when the process exits. This avoids re-embedding a document that
# was already indexed during the same run (e.g., if the same PDF is queried
# for both confidentiality AND termination clauses in one session).

_embedding_cache: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings in ONE batched API call.

    Why one call?
        Each OpenAI embedding API call has network latency. Sending 200
        chunks one-by-one would be 200 round trips. Sending them all in
        one request costs the same in tokens but takes the same time as
        one round trip.

    Returns
    -------
    np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype float32.
    Vectors are L2-normalized so that inner-product == cosine similarity.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    from agent.metrics_writer import accumulate_tokens
    accumulate_tokens(embedding=response.usage.total_tokens)

    # response.data is a list of Embedding objects, in the same order as `texts`
    vectors = np.array(
        [item.embedding for item in response.data],
        dtype=np.float32,
    )

    # L2-normalize so IndexFlatIP gives cosine similarity scores
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    vectors = vectors / norms

    return vectors


def embed_queries(queries: list[str]) -> np.ndarray:
    """
    Embed one or more query strings in a single batched call.

    Identical to embed_texts -- kept as a separate function so callers can
    clearly distinguish "I am embedding document content" vs "I am embedding
    a search query". Same underlying call, clearer intent.
    """
    return embed_texts(queries)


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------

def _mmr_select(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_indices: np.ndarray,
    top_n: int,
    lambda_: float = MMR_LAMBDA,
) -> list[int]:
    """
    Apply Maximal Marginal Relevance to select top_n diverse candidates.

    Parameters
    ----------
    query_vector : np.ndarray shape (dim,)
    candidate_vectors : np.ndarray shape (num_candidates, dim)
    candidate_indices : np.ndarray shape (num_candidates,)
        Original FAISS indices corresponding to each candidate vector.
    top_n : int
    lambda_ : float
        Trade-off weight. 1.0 = pure relevance, 0.0 = pure diversity.

    Returns
    -------
    list[int]
        Ordered list of selected indices into the original FAISS index.
    """
    selected: list[int] = []
    remaining = list(range(len(candidate_vectors)))

    # Relevance scores: cosine similarity to query (already normalized vectors)
    rel_scores = candidate_vectors @ query_vector  # shape: (num_candidates,)

    while len(selected) < top_n and remaining:
        if not selected:
            # First pick: highest relevance, no diversity penalty yet
            best_local = int(np.argmax(rel_scores[remaining]))
        else:
            selected_vecs = candidate_vectors[selected]  # shape: (|selected|, dim)
            best_score = -float("inf")
            best_local = remaining[0]
            for idx in remaining:
                rel = lambda_ * rel_scores[idx]
                # Diversity: penalise similarity to already-selected items
                sim_to_selected = float(np.max(candidate_vectors[idx] @ selected_vecs.T))
                div = (1 - lambda_) * sim_to_selected
                score = rel - div
                if score > best_score:
                    best_score = score
                    best_local = idx

        selected.append(best_local)
        remaining.remove(best_local)

    return [int(candidate_indices[i]) for i in selected]


# ---------------------------------------------------------------------------
# LLM re-ranker
# ---------------------------------------------------------------------------

def _rerank_with_llm(query: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    Ask GPT-4o-mini to re-order `chunks` by relevance to `query`.

    Returns the same chunks list re-ordered (best first). If the LLM call
    fails for any reason, the original FAISS order is returned unchanged.

    Why GPT-4o-mini and not a dedicated cross-encoder?
        A cross-encoder (like a small BERT re-ranker) would be faster and
        cheaper per call, but would need a separate model download and
        inference setup. GPT-4o-mini is already in the stack and gives
        high-quality legal-domain relevance judgments. We only call it when
        the top-1 similarity is ambiguous (< 0.92), so cost is low.
    """
    if not chunks:
        return chunks

    # Build a numbered list of chunk excerpts for the model to judge
    numbered = "\n\n".join(
        f"[{i+1}] {chunk.text[:400]}{'...' if len(chunk.text) > 400 else ''}"
        for i, chunk in enumerate(chunks)
    )

    prompt = (
        f"You are a legal document retrieval assistant. "
        f"A user is looking for information about: \"{query}\"\n\n"
        f"Below are {len(chunks)} candidate text excerpts from a legal document, "
        f"numbered 1 to {len(chunks)}.\n\n"
        f"{numbered}\n\n"
        f"Rank these excerpts from most relevant to least relevant for the user's query. "
        f"Output ONLY a JSON array of the excerpt numbers in your preferred order, "
        f"from most to least relevant. Example: [3, 1, 2]. "
        f"Output the JSON array and nothing else."
    )

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,
        )
        from agent.metrics_writer import accumulate_tokens
        accumulate_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
        raw = (response.choices[0].message.content or "").strip()
        # Parse JSON array like [3, 1, 2]
        order = json.loads(raw)
        if isinstance(order, list) and all(isinstance(x, int) for x in order):
            # Convert 1-based to 0-based and filter valid indices
            reordered = []
            seen = set()
            for rank in order:
                idx = rank - 1
                if 0 <= idx < len(chunks) and idx not in seen:
                    reordered.append(chunks[idx])
                    seen.add(idx)
            # Append any chunks the LLM omitted (shouldn't happen, but defensive)
            for i, chunk in enumerate(chunks):
                if i not in seen:
                    reordered.append(chunk)
            logger.debug(f"LLM re-ranked {len(chunks)} chunks; new order: {order}")
            return reordered
    except Exception as e:
        logger.warning(f"LLM re-rank failed ({e}); falling back to FAISS order.")

    return chunks


# ---------------------------------------------------------------------------
# VectorStore class
# ---------------------------------------------------------------------------

class VectorStore:
    """
    FAISS-backed vector store for a collection of DocumentChunks.

    Typical lifecycle:
        store = VectorStore()
        store.add_chunks(chunks)           # embed + index
        store.save(path)                   # persist to disk
        store = VectorStore.load(path)     # restore later
        results = store.search(query)      # retrieve top-3 chunks

    One VectorStore instance per document (or per document session).
    For comparing two contracts, create two stores (or one combined store
    and tag chunks with their source document hash).
    """

    def __init__(self) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._chunks: list[DocumentChunk] = []
        self._vectors: Optional[np.ndarray] = None  # kept for MMR

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Embed all chunks in a single batched API call and add them to the index.

        Per-session cache: if a document's chunks have already been embedded
        in this process run (same document_hash), the cached vectors are used
        directly without a new API call.
        """
        if not chunks:
            logger.warning("add_chunks called with an empty list.")
            return

        doc_hash = chunks[0].document_hash

        if doc_hash in _embedding_cache:
            logger.info(
                f"Embedding cache hit for '{chunks[0].document_name}' "
                f"({doc_hash[:12]}...) -- skipping API call."
            )
            vectors = _embedding_cache[doc_hash]
        else:
            logger.info(
                f"Embedding {len(chunks)} chunks for '{chunks[0].document_name}' "
                f"via {EMBEDDING_MODEL} (one batched call)..."
            )
            texts = [chunk.text for chunk in chunks]
            vectors = embed_texts(texts)
            _embedding_cache[doc_hash] = vectors
            logger.info(f"Embedding complete. Shape: {vectors.shape}")

        import faiss
        self._index.add(vectors)
        self._chunks.extend(chunks)

        # Maintain a local copy of all stored vectors for MMR
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """
        Persist the FAISS index and chunk metadata to `directory`.

        Creates two files:
            index.faiss  -- the FAISS index binary
            chunks.pkl   -- the list of DocumentChunk objects
        """
        import faiss
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"VectorStore saved to '{path}' ({len(self._chunks)} chunks).")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """
        Load a previously saved VectorStore from disk.
        """
        import faiss
        path = Path(directory)
        store = cls()
        store._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            store._chunks = pickle.load(f)
        # Reconstruct the local vector matrix for MMR
        # (We re-embed from the index's internal storage)
        # Note: IndexFlatIP stores vectors internally; we reconstruct via reconstruct_n
        n = store._index.ntotal
        if n > 0:
            vecs = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
            store._index.reconstruct_n(0, n, vecs)
            store._vectors = vecs
        logger.info(f"VectorStore loaded from '{path}' ({len(store._chunks)} chunks).")
        return store

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_n: int = RERANK_TOP_N,
        use_mmr: bool = True,
        use_rerank: bool = True,
    ) -> list[DocumentChunk]:
        """
        Search the index for chunks relevant to `query`.

        Steps:
            1. Embed the query.
            2. Run FAISS to get top FAISS_TOP_K candidates.
            3. Apply MMR to pick the most relevant+diverse subset.
            4. If top-1 cosine similarity < RERANK_SKIP_THRESHOLD, re-rank
               with GPT-4o-mini.
            5. Return top_n chunks.

        Parameters
        ----------
        query : str
            Natural-language question or clause type to search for.
        top_n : int
            Number of chunks to return (default 3, per locked architecture).
        use_mmr : bool
            If False, return raw FAISS results (useful for ablation tests).
        use_rerank : bool
            If False, skip LLM re-ranking (useful for offline/no-API testing).

        Returns
        -------
        list[DocumentChunk]
            Ordered best-first. Empty list if the index has no entries.
        """
        if self._index.ntotal == 0:
            logger.warning("search() called on an empty VectorStore.")
            return []

        # Step 1: Embed the query
        query_vec = embed_queries([query])[0]  # shape: (EMBEDDING_DIM,)

        # Step 2: FAISS top-K
        k = min(FAISS_TOP_K, self._index.ntotal)
        scores, faiss_indices = self._index.search(
            query_vec.reshape(1, -1), k
        )
        scores = scores[0]        # shape: (k,)
        faiss_indices = faiss_indices[0]  # shape: (k,)

        top1_similarity = float(scores[0]) if len(scores) > 0 else 0.0

        logger.debug(
            f"FAISS top-1 cosine similarity: {top1_similarity:.4f} "
            f"(rerank threshold={RERANK_SKIP_THRESHOLD})"
        )

        # Step 3: MMR selection
        if use_mmr and self._vectors is not None and len(faiss_indices) > top_n:
            candidate_vecs = self._vectors[faiss_indices]
            selected_faiss_idx = _mmr_select(
                query_vector=query_vec,
                candidate_vectors=candidate_vecs,
                candidate_indices=faiss_indices,
                top_n=min(FAISS_TOP_K, len(faiss_indices)),
            )
            selected_chunks = [self._chunks[i] for i in selected_faiss_idx]
        else:
            selected_chunks = [self._chunks[i] for i in faiss_indices]

        # Step 4: LLM re-rank (skip if top-1 is a clear winner)
        if (
            use_rerank
            and len(selected_chunks) > 1
            and top1_similarity < RERANK_SKIP_THRESHOLD
        ):
            logger.debug(
                f"Top-1 similarity {top1_similarity:.4f} < {RERANK_SKIP_THRESHOLD} "
                f"-- calling LLM re-ranker."
            )
            selected_chunks = _rerank_with_llm(query, selected_chunks)
        elif top1_similarity >= RERANK_SKIP_THRESHOLD:
            logger.debug(
                f"Top-1 similarity {top1_similarity:.4f} >= {RERANK_SKIP_THRESHOLD} "
                f"-- skipping LLM re-rank (clear winner)."
            )

        return selected_chunks[:top_n]

    def search_batch(
        self,
        queries: list[str],
        top_n: int = RERANK_TOP_N,
        use_mmr: bool = True,
        use_rerank: bool = True,
    ) -> dict[str, list[DocumentChunk]]:
        """
        Search for multiple queries in a SINGLE batched embedding call.

        Per CLAUDE.md: "batch all clause queries into one embedding call
        rather than 10 separate ones."

        Returns
        -------
        dict mapping each query string to its list of top_n DocumentChunks.
        """
        if not queries:
            return {}

        if self._index.ntotal == 0:
            return {q: [] for q in queries}

        # Embed all queries at once
        query_vecs = embed_queries(queries)  # shape: (num_queries, EMBEDDING_DIM)

        k = min(FAISS_TOP_K, self._index.ntotal)
        all_scores, all_indices = self._index.search(query_vecs, k)

        results: dict[str, list[DocumentChunk]] = {}

        for qi, query in enumerate(queries):
            scores = all_scores[qi]
            faiss_indices = all_indices[qi]
            top1_similarity = float(scores[0]) if len(scores) > 0 else 0.0

            if use_mmr and self._vectors is not None and len(faiss_indices) > top_n:
                candidate_vecs = self._vectors[faiss_indices]
                selected_idx = _mmr_select(
                    query_vector=query_vecs[qi],
                    candidate_vectors=candidate_vecs,
                    candidate_indices=faiss_indices,
                    top_n=min(FAISS_TOP_K, len(faiss_indices)),
                )
                selected_chunks = [self._chunks[i] for i in selected_idx]
            else:
                selected_chunks = [self._chunks[i] for i in faiss_indices]

            if (
                use_rerank
                and len(selected_chunks) > 1
                and top1_similarity < RERANK_SKIP_THRESHOLD
            ):
                selected_chunks = _rerank_with_llm(query, selected_chunks)

            results[query] = selected_chunks[:top_n]

        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"VectorStore(chunks={self.chunk_count}, indexed={self._index.ntotal})"
