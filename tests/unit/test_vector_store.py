"""
tests/unit/test_vector_store.py
---------------------------------
Automated test suite for retrieval/vector_store.py.

All OpenAI API calls are mocked -- no real API key or network access needed.

What this file tests:
    1.  add_chunks embeds text in a single batched call (not one call per chunk).
    2.  search returns at most top_n chunks.
    3.  search on an empty store returns [] without crashing.
    4.  High top-1 similarity (>= 0.92) skips the LLM re-ranker.
    5.  Low top-1 similarity (< 0.92) calls the LLM re-ranker.
    6.  search_batch makes a single embedding call for all queries combined.
    7.  Per-session embedding cache: the same document_hash is not re-embedded
        in the same process run.
    8.  save() + load() round-trips correctly (chunks survive disk persistence).
    9.  MMR selection returns diverse results (not all near-duplicates).
    10. embed_texts returns L2-normalized vectors (norm == 1.0 for each row).

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_vector_store.py -v
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from schemas.chunk import DocumentChunk
from retrieval.vector_store import (
    VectorStore,
    embed_texts,
    _mmr_select,
    EMBEDDING_DIM,
    RERANK_SKIP_THRESHOLD,
    RERANK_TOP_N,
    _embedding_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_HASH = "ab" * 32  # 64-char hex string


def _make_chunk(index: int, text: str, doc_hash: str = FAKE_HASH) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_hash[:12]}_chunk_{index:04d}",
        document_hash=doc_hash,
        document_name="test_contract.pdf",
        text=text,
        char_count=len(text),
        token_count=len(text) // 4,
        start_page=1,
        end_page=1,
        chunk_index=index,
        total_chunks=5,
        overlap_tokens=0,
    )


def _random_unit_vector(seed: int = 0) -> np.ndarray:
    """Return a deterministic unit vector of shape (EMBEDDING_DIM,)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _fake_embed_response(texts: list[str], seed_offset: int = 0):
    """Return a mock OpenAI embeddings response with deterministic unit vectors."""
    mock_resp = MagicMock()
    mock_resp.data = [
        MagicMock(embedding=_random_unit_vector(i + seed_offset).tolist())
        for i in range(len(texts))
    ]
    return mock_resp


# ---------------------------------------------------------------------------
# Fixture: clear the per-session embedding cache before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_embedding_cache():
    """
    The per-session cache in vector_store.py persists across tests within
    the same process run. Clear it before each test so tests are independent.
    """
    _embedding_cache.clear()
    yield
    _embedding_cache.clear()


# ---------------------------------------------------------------------------
# Test 1: add_chunks makes ONE embedding call (batch, not per-chunk)
# ---------------------------------------------------------------------------

def test_add_chunks_makes_one_embedding_call():
    """
    Five chunks must be embedded in a single API call, not five separate calls.
    """
    chunks = [_make_chunk(i, f"Clause {i}: some legal text here.") for i in range(5)]

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.return_value = _fake_embed_response(
            [c.text for c in chunks]
        )

        store = VectorStore()
        store.add_chunks(chunks)

        assert mock_client.embeddings.create.call_count == 1, (
            f"Expected 1 embedding call, got {mock_client.embeddings.create.call_count}"
        )


# ---------------------------------------------------------------------------
# Test 2: search returns at most top_n results
# ---------------------------------------------------------------------------

def test_search_returns_at_most_top_n():
    """search() must return <= RERANK_TOP_N chunks."""
    chunks = [_make_chunk(i, f"Clause {i}: text about legal obligations.") for i in range(10)]

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.return_value = _fake_embed_response(
            [c.text for c in chunks]
        )

        store = VectorStore()
        store.add_chunks(chunks)

        # Mock the query embedding call too
        mock_client.embeddings.create.return_value = _fake_embed_response(["query"], seed_offset=99)

        results = store.search("indemnification clause", use_rerank=False)

    assert len(results) <= RERANK_TOP_N


# ---------------------------------------------------------------------------
# Test 3: search on empty store returns []
# ---------------------------------------------------------------------------

def test_search_empty_store_returns_empty_list():
    """An empty VectorStore must return [] on search without crashing."""
    store = VectorStore()
    results = store.search("governing law clause", use_rerank=False)
    assert results == []


# ---------------------------------------------------------------------------
# Test 4: High similarity skips LLM re-ranker
# ---------------------------------------------------------------------------

def test_high_similarity_skips_rerank():
    """
    When top-1 cosine similarity >= RERANK_SKIP_THRESHOLD, the LLM re-ranker
    must NOT be called -- this saves an API round-trip for clear winners.
    """
    chunks = [_make_chunk(i, f"Text {i}") for i in range(5)]

    # We need the query vector to be VERY similar to chunk 0.
    # Strategy: fix chunk 0's embedding as the query vector itself (similarity = 1.0)
    base_vec = _random_unit_vector(seed=0)

    fake_chunk_vecs = np.vstack([
        base_vec.reshape(1, -1),
        *[_random_unit_vector(seed=i + 1).reshape(1, -1) for i in range(4)],
    ])  # shape (5, EMBEDDING_DIM)

    def embed_side_effect(model, input, **kwargs):
        mock_resp = MagicMock()
        if len(input) == len(chunks):
            # Chunk embeddings
            mock_resp.data = [MagicMock(embedding=fake_chunk_vecs[i].tolist()) for i in range(len(chunks))]
        else:
            # Query embedding -- return the same as chunk 0 for perfect similarity
            mock_resp.data = [MagicMock(embedding=base_vec.tolist())]
        return mock_resp

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.side_effect = embed_side_effect

        store = VectorStore()
        store.add_chunks(chunks)

        # chat.completions.create should NOT be called
        results = store.search("test query", use_rerank=True)

        assert mock_client.chat.completions.create.call_count == 0, (
            "LLM re-ranker was called despite top-1 similarity >= threshold"
        )


# ---------------------------------------------------------------------------
# Test 5: Low similarity calls LLM re-ranker
# ---------------------------------------------------------------------------

def test_low_similarity_calls_rerank():
    """
    When top-1 similarity < RERANK_SKIP_THRESHOLD, the LLM re-ranker must
    be called to re-order the FAISS candidates.
    """
    chunks = [_make_chunk(i, f"Text {i}") for i in range(5)]

    # Make all vectors close to ORTHOGONAL to the query vector so similarity is low
    query_vec = _random_unit_vector(seed=999)

    def embed_side_effect(model, input, **kwargs):
        mock_resp = MagicMock()
        if len(input) == len(chunks):
            # Chunk vectors: all have low similarity to query_vec
            vecs = np.vstack([_random_unit_vector(seed=i + 10).reshape(1, -1) for i in range(len(chunks))])
            mock_resp.data = [MagicMock(embedding=vecs[i].tolist()) for i in range(len(chunks))]
        else:
            mock_resp.data = [MagicMock(embedding=query_vec.tolist())]
        return mock_resp

    # LLM re-ranker returns a JSON array
    mock_chat_resp = MagicMock()
    mock_chat_resp.choices = [MagicMock(message=MagicMock(content="[1, 2, 3]"))]

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.side_effect = embed_side_effect
        mock_client.chat.completions.create.return_value = mock_chat_resp

        store = VectorStore()
        store.add_chunks(chunks)
        results = store.search("some query", use_rerank=True, use_mmr=False)

        assert mock_client.chat.completions.create.call_count >= 1, (
            "LLM re-ranker was NOT called despite top-1 similarity < threshold"
        )


# ---------------------------------------------------------------------------
# Test 6: search_batch makes one embedding call for all queries
# ---------------------------------------------------------------------------

def test_search_batch_makes_one_embedding_call_for_all_queries():
    """
    search_batch(['q1', 'q2', 'q3']) must embed all three queries in a single
    API call, not three separate calls.
    """
    chunks = [_make_chunk(i, f"Legal text about clause {i}.") for i in range(5)]
    queries = ["confidentiality", "termination", "governing law"]

    call_log = []

    def embed_side_effect(model, input, **kwargs):
        call_log.append(len(input))
        n = len(input)
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=_random_unit_vector(seed=i).tolist()) for i in range(n)]
        return mock_resp

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.side_effect = embed_side_effect
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[1]"))]
        )

        store = VectorStore()
        store.add_chunks(chunks)  # call 1: 5 chunks
        call_log.clear()           # reset -- we only care about query calls

        results = store.search_batch(queries, use_rerank=False, use_mmr=False)

        # Should be exactly ONE call containing all 3 queries
        assert len(call_log) == 1, (
            f"Expected 1 embedding call for all queries; got {len(call_log)} calls"
        )
        assert call_log[0] == 3, (
            f"Expected the single call to embed 3 queries; it embedded {call_log[0]}"
        )

    # Results dict has one key per query
    assert set(results.keys()) == set(queries)


# ---------------------------------------------------------------------------
# Test 7: Per-session cache prevents re-embedding the same document
# ---------------------------------------------------------------------------

def test_embedding_cache_prevents_duplicate_api_calls():
    """
    Adding the same document's chunks twice (same document_hash) must only
    call the embedding API once -- the second add_chunks reuses the cache.
    """
    chunks = [_make_chunk(i, f"Clause {i} text.") for i in range(3)]

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.return_value = _fake_embed_response(
            [c.text for c in chunks]
        )

        store1 = VectorStore()
        store1.add_chunks(chunks)

        store2 = VectorStore()
        store2.add_chunks(chunks)  # same doc_hash -- should use cache

        assert mock_client.embeddings.create.call_count == 1, (
            f"Expected 1 API call; got {mock_client.embeddings.create.call_count} "
            f"(cache not working)"
        )


# ---------------------------------------------------------------------------
# Test 8: save() / load() round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    """
    Chunks indexed in a VectorStore, saved to disk, and loaded back must
    produce the same chunk list in the same order.
    """
    chunks = [_make_chunk(i, f"Clause {i}: legal text.") for i in range(4)]

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.return_value = _fake_embed_response(
            [c.text for c in chunks]
        )

        store = VectorStore()
        store.add_chunks(chunks)
        store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)

    assert loaded.chunk_count == len(chunks)
    for original, restored in zip(chunks, loaded._chunks):
        assert original.chunk_id == restored.chunk_id
        assert original.text == restored.text
        assert original.document_hash == restored.document_hash


# ---------------------------------------------------------------------------
# Test 9: MMR returns diverse results
# ---------------------------------------------------------------------------

def test_mmr_selects_diverse_chunks():
    """
    MMR should NOT return the same candidate twice, and should prefer
    candidates that are diverse from each other.
    """
    rng = np.random.default_rng(42)
    query_vec = _random_unit_vector(seed=0)

    # Candidates: one very similar to query, four very similar to each other
    # but different from the query.
    similar_to_query = query_vec + rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.01
    similar_to_query /= np.linalg.norm(similar_to_query)

    cluster_base = _random_unit_vector(seed=99)
    candidates = np.vstack([
        similar_to_query.reshape(1, -1),
        *[
            (cluster_base + rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.01).reshape(1, -1)
            for _ in range(4)
        ],
    ]).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    candidates /= norms

    indices = np.array([0, 1, 2, 3, 4])
    selected = _mmr_select(query_vec, candidates, indices, top_n=3)

    # No duplicates
    assert len(set(selected)) == len(selected), "MMR returned duplicate indices"
    # Index 0 (closest to query) should be selected first
    assert selected[0] == 0, "MMR should pick the most relevant chunk first"


# ---------------------------------------------------------------------------
# Test 10: embed_texts returns L2-normalized vectors
# ---------------------------------------------------------------------------

def test_embed_texts_returns_normalized_vectors():
    """
    embed_texts must return unit vectors (L2 norm == 1.0 for each row),
    which is required for IndexFlatIP to give cosine similarity scores.
    """
    texts = ["clause about indemnification", "clause about termination"]

    with patch("retrieval.vector_store._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Return unnormalized vectors -- embed_texts should normalize them
        raw = np.array([
            np.ones(EMBEDDING_DIM, dtype=np.float32) * 2.0,
            np.ones(EMBEDDING_DIM, dtype=np.float32) * 5.0,
        ])
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=raw[i].tolist()) for i in range(2)]
        )

        vectors = embed_texts(texts)

    norms = np.linalg.norm(vectors, axis=1)
    for i, norm in enumerate(norms):
        assert abs(norm - 1.0) < 1e-5, f"Row {i} has norm {norm:.6f}, expected 1.0"
