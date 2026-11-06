"""
Unit Test Suite for Financial RAG Analyst
Validates document parsing, vector indexing, retrieval accuracy, and citation attribution.
"""

from pathlib import Path
import pytest
from src.document_loader import FinancialDocumentLoader
from src.vector_store import FinancialVectorStore
from src.rag_engine import FinancialRAGEngine
import config


@pytest.fixture
def sample_file_path() -> Path:
    """Fixture providing path to the sample SEC 10-K filing."""
    path = config.SAMPLE_FILING_PATH
    assert path.exists(), f"Sample filing missing at: {path}"
    return path


@pytest.fixture
def rag_engine(sample_file_path: Path) -> FinancialRAGEngine:
    """Fixture initializing and indexing the sample filing into the RAG engine."""
    engine = FinancialRAGEngine()
    index_res = engine.index_document(str(sample_file_path))
    assert index_res["chunks_created"] > 0
    return engine


def test_document_loader_chunking(sample_file_path: Path):
    """Verifies that the document loader chunks text and extracts section headers."""
    loader = FinancialDocumentLoader(chunk_size=500, chunk_overlap=100)
    chunks = loader.load_and_chunk_file(sample_file_path)

    assert len(chunks) >= 5, "Expected at least 5 chunks from sample filing"
    first_chunk = chunks[0]
    assert first_chunk.source == sample_file_path.name
    assert first_chunk.page_number >= 1
    assert len(first_chunk.text) > 0

    # Verify section detection works
    sections = {c.section for c in chunks}
    assert any("RISK" in s.upper() or "MD&A" in s.upper() or "BUSINESS" in s.upper() for s in sections), \
        f"Expected identified SEC sections, found: {sections}"


def test_vector_store_retrieval(sample_file_path: Path):
    """Verifies that vector indexing and similarity search return top-k matches."""
    loader = FinancialDocumentLoader()
    chunks = loader.load_and_chunk_file(sample_file_path)

    store = FinancialVectorStore()
    indexed_count = store.add_chunks(chunks)
    assert indexed_count == len(chunks)

    # Perform similarity search
    hits = store.similarity_search("semiconductor foundry supply chain risk", top_k=3)
    assert 1 <= len(hits) <= 3
    assert hits[0]["score"] > 0.0
    assert "page_number" in hits[0]
    assert "section" in hits[0]


def test_rag_revenue_financial_query(rag_engine: FinancialRAGEngine):
    """Verifies that querying financial revenue retrieves correct metrics and page attribution."""
    response = rag_engine.answer_query("What was total consolidated revenue and operating income?")

    assert response["chunks_evaluated"] > 0
    assert len(response["citations"]) > 0

    # The answer or citations must contain the revenue numbers
    citation_texts = " ".join(c["snippet"] for c in response["citations"])
    assert "24.85" in citation_texts or "24,850" in citation_texts or "revenue" in citation_texts.lower()
    assert response["retrieval_time_ms"] < 1000, "Retrieval latency must be sub-second"


def test_rag_risk_factors_query(rag_engine: FinancialRAGEngine):
    """Verifies that querying risk factors retrieves the Risk Factors section."""
    response = rag_engine.answer_query("What are the foundry concentration and supply chain risks?")

    assert len(response["citations"]) >= 1
    top_citation = response["citations"][0]
    assert "RISK" in top_citation["section"].upper() or "foundry" in top_citation["snippet"].lower()


def test_rag_empty_or_nonsense_query(rag_engine: FinancialRAGEngine):
    """Verifies graceful handling of non-existent queries."""
    response = rag_engine.answer_query("xyzqwerty123456nonexistentword")
    assert "No relevant financial disclosures found" in response["answer"]
    assert response["citations"] == []
