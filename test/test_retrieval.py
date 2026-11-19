"""Retriever tests. Uses a real temporary ChromaDB with a fake embedding
model, so no network or model download is needed."""

import hashlib
import math

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.metadata_tagger import tag_chunks
from rag.retrieval import  (
    HybridRetriever,
    MetadataFilter,
    RetrieverConfig,
    only_superseded,
    _tokenize,
    reciprocal_rank_fusion,
)


class FakeEmbeddings(Embeddings):
    """Hashed bag-of-words vectors: similar words give similar vectors."""

    DIM = 64

    def _embed(self, text):
        vector = [0.0] * self.DIM
        for word in text.lower().split():
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            vector[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def make_chunk(chunk_id, text, **meta):
    base = {"chunk_id": chunk_id, "source": "pack.pdf", "page": 1}
    base.update(meta)
    return Document(page_content=text, metadata=base)


@pytest.fixture
def store(tmp_path):
    chunks = [
        make_chunk("a", "Business Plus monthly fee is 650 rupees waived at 50000 balance",
                   version="2026", status="current", doc_family="fee_schedule",
                   content_type="table"),
        make_chunk("b", "Business Plus monthly fee is 575 rupees waived at 40000 balance",
                   version="2025", status="superseded", doc_family="fee_schedule",
                   content_type="table"),
        make_chunk("c", "Sole trader KYC needs photo ID and address proof",
                   version="2026", status="current", doc_family="kyc_aml",
                   content_type="text"),
        make_chunk("d", "Overdraft policy 2025 note limits assigned after review",
                   version="2025", status="superseded", doc_family="product_terms",
                   content_type="text"),
    ]
    vector_store = Chroma(
        collection_name="test_docs",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    vector_store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return vector_store


# ---------- RRF ----------

def test_rrf_rewards_chunks_found_by_both_lists():
    scores = reciprocal_rank_fusion([["x", "y"], ["y", "z"]], rrf_k=60)
    assert scores["y"] > scores["x"]
    assert scores["y"] > scores["z"]


def test_rrf_weight_zero_ignores_a_list():
    scores = reciprocal_rank_fusion([["x"], ["y"]], weights=[1.0, 0.0])
    assert scores["y"] == 0.0


# ---------- filters ----------

def test_filter_to_chroma_shapes():
    assert MetadataFilter().to_chroma() is None
    assert MetadataFilter(status="current").to_chroma() == {"status": "current"}
    assert MetadataFilter(version=["2025", "2026"]).to_chroma() == {
        "version": {"$in": ["2025", "2026"]}
    }
    both = MetadataFilter(status="current", doc_family="kyc_aml").to_chroma()
    assert both == {"$and": [{"status": "current"}, {"doc_family": "kyc_aml"}]}


def test_filter_rejects_empty_list():
    with pytest.raises(ValueError):
        MetadataFilter(status=[])


def test_filter_matches_metadata():
    flt = MetadataFilter(status="current", version=["2025", "2026"])
    assert flt.matches({"status": "current", "version": "2026"})
    assert not flt.matches({"status": "superseded", "version": "2026"})


def test_tokenizer_joins_indian_and_western_number_groups():
    assert _tokenize("Waiver at ₹2,00,000 or 50,000") == ["waiver", "at", "200000", "or", "50000"]


# ---------- retrieval ----------

def test_exact_number_found_by_keyword_side(store):
    retriever = HybridRetriever(store, RetrieverConfig(top_k=2))
    results = retriever.retrieve("575")
    assert results[0].document.metadata["chunk_id"] == "b"
    assert results[0].bm25_rank == 1


def test_status_filter_applies_to_both_searches(store):
    retriever = HybridRetriever(store, RetrieverConfig(top_k=4))
    results = retriever.retrieve(
        "Business Plus monthly fee", MetadataFilter(status="current")
    )
    assert results
    assert all(r.document.metadata["status"] == "current" for r in results)


def test_combined_filters(store):
    retriever = HybridRetriever(store)
    results = retriever.retrieve(
        "fee", MetadataFilter(doc_family="fee_schedule", content_type="table",
                              status="superseded")
    )
    assert [r.document.metadata["chunk_id"] for r in results] == ["b"]


def test_top_k_and_candidate_limits(store):
    retriever = HybridRetriever(
        store, RetrieverConfig(top_k=10, dense_candidates=2, bm25_candidates=1)
    )
    results = retriever.retrieve("Business Plus monthly fee")
    assert len(results) <= 3  # at most 2 dense + 1 bm25 candidates
    assert len(retriever.retrieve("Business Plus fee", top_k=1)) == 1


def test_results_are_deterministic(store):
    retriever = HybridRetriever(store)
    first = [r.document.metadata["chunk_id"] for r in retriever.retrieve("fee")]
    second = [r.document.metadata["chunk_id"] for r in retriever.retrieve("fee")]
    assert first == second


def test_filter_with_no_match_returns_empty(store):
    retriever = HybridRetriever(store)
    assert retriever.retrieve("fee", MetadataFilter(doc_family="nothing")) == []


def test_only_superseded_flag(store):
    retriever = HybridRetriever(store)
    old = retriever.retrieve("fee", MetadataFilter(status="superseded"))
    new = retriever.retrieve("fee", MetadataFilter(status="current"))
    assert only_superseded(old)
    assert not only_superseded(new)
    assert not only_superseded([])


def test_empty_query_and_bad_k_rejected(store):
    retriever = HybridRetriever(store)
    with pytest.raises(ValueError):
        retriever.retrieve("   ")
    with pytest.raises(ValueError):
        retriever.retrieve("fee", top_k=0)
    with pytest.raises(ValueError):
        RetrieverConfig(top_k=0)


def test_langchain_adapter(store):
    lc = HybridRetriever(store).as_langchain_retriever(MetadataFilter(status="current"))
    docs = lc.invoke("Sole trader KYC documents")
    assert docs
    assert "rrf_score" in docs[0].metadata


def test_refresh_index_picks_up_new_chunks(store):
    retriever = HybridRetriever(store)
    assert all(r.bm25_rank is None for r in retriever.retrieve("zebra"))

    new_chunk = make_chunk("z", "zebra crossing note", version="2026", status="current")
    store.add_documents([new_chunk], ids=["z"])
    retriever.refresh_index()

    top = retriever.retrieve("zebra")[0]
    assert top.document.metadata["chunk_id"] == "z"
    assert top.bm25_rank == 1


# ---------- tagger ----------

def test_tagger_family_carries_across_chunks_and_resets_per_document():
    chunks = [
        make_chunk("1", "6. Fee schedule - account maintenance\nBusiness Plus", version="2026"),
        make_chunk("2", "Waiver condition\nMaintain balance", version="2026"),
        make_chunk("3", "4. KYC/AML manual - onboarding\n1. Identity\n2. Address", version="2026"),
        make_chunk("4", "Some text with no heading", version="2025"),
    ]
    tag_chunks(chunks)
    families = [c.metadata["doc_family"] for c in chunks]
    assert families == ["fee_schedule", "fee_schedule", "kyc_aml", "general"]
    assert chunks[0].metadata["effective_date"] == "2026-01-01"
    assert chunks[3].metadata["effective_date"] == "2025-01-01"


def test_tagger_ignores_numbered_table_rows():
    chunk = make_chunk("1", "1. Identity\n2. Address\n3. Business", version="2026")
    tag_chunks([chunk])
    assert chunk.metadata["doc_family"] == "general"


def test_tagger_content_types():
    table = make_chunk("1", "| a | b |\n|---|---|\n| 1 | 2 |")
    image = make_chunk("2", "Diagram\n<!-- image -->")
    prose = make_chunk("3", "This is a normal sentence about the account.")
    tag_chunks([table, image, prose])
    assert table.metadata["content_type"] == "table"
    assert image.metadata["content_type"] == "image"
    assert prose.metadata["content_type"] == "text"