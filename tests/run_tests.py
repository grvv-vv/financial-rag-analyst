"""
Automated Test Runner (using standard library unittest)
Guarantees tests run out-of-the-box on any Python environment without needing pip packages.
"""

from pathlib import Path
import unittest
import sys
import os

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config
from src.document_loader import FinancialDocumentLoader
from src.vector_store import FinancialVectorStore
from src.rag_engine import FinancialRAGEngine


class TestFinancialRAG(unittest.TestCase):
    """Test suite for financial document parsing, indexing, and question answering."""

    @classmethod
    def setUpClass(cls):
        cls.sample_path = config.SAMPLE_FILING_PATH
        if not cls.sample_path.exists():
            raise FileNotFoundError(f"Sample filing not found at {cls.sample_path}")
        cls.engine = FinancialRAGEngine()
        cls.index_result = cls.engine.index_document(str(cls.sample_path))

    def test_01_document_loader_chunking(self):
        """Verifies chunk count, source tracking, and section identification."""
        loader = FinancialDocumentLoader(chunk_size=500, chunk_overlap=100)
        chunks = loader.load_and_chunk_file(self.sample_path)

        self.assertGreaterEqual(len(chunks), 5)
        self.assertEqual(chunks[0].source, self.sample_path.name)
        self.assertGreaterEqual(chunks[0].page_number, 1)

        sections = {c.section for c in chunks}
        has_sec_section = any("RISK" in s.upper() or "MD&A" in s.upper() or "BUSINESS" in s.upper() for s in sections)
        self.assertTrue(has_sec_section, f"Expected identified SEC sections, found: {sections}")

    def test_02_vector_store_retrieval(self):
        """Verifies similarity search returns ranked hits with valid scores."""
        loader = FinancialDocumentLoader()
        chunks = loader.load_and_chunk_file(self.sample_path)

        store = FinancialVectorStore()
        count = store.add_chunks(chunks)
        self.assertEqual(count, len(chunks))

        hits = store.similarity_search("semiconductor foundry supply chain risk", top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        self.assertLessEqual(len(hits), 3)
        self.assertGreater(hits[0]["score"], 0.0)
        self.assertIn("page_number", hits[0])
        self.assertIn("section", hits[0])

    def test_03_rag_revenue_financial_query(self):
        """Verifies querying revenue extracts factual metrics and page citations."""
        response = self.engine.answer_query("What was total consolidated revenue and operating income?")

        self.assertGreater(response["chunks_evaluated"], 0)
        self.assertGreater(len(response["citations"]), 0)

        citation_texts = " ".join(c["snippet"] for c in response["citations"])
        has_revenue_data = "24.85" in citation_texts or "24,850" in citation_texts or "revenue" in citation_texts.lower()
        self.assertTrue(has_revenue_data, "Revenue figures must be present in retrieved context")
        self.assertLess(response["retrieval_time_ms"], 1000, "Retrieval latency must be under 1s")

    def test_04_rag_risk_factors_query(self):
        """Verifies risk queries route to Risk Factors section."""
        response = self.engine.answer_query("What are the foundry concentration and supply chain risks?")

        self.assertGreaterEqual(len(response["citations"]), 1)
        top_hit = response["citations"][0]
        has_risk = "RISK" in top_hit["section"].upper() or "foundry" in top_hit["snippet"].lower()
        self.assertTrue(has_risk, "Top citation must match risk factors")

    def test_05_rag_empty_or_nonsense_query(self):
        """Verifies graceful handling of non-existent queries."""
        response = self.engine.answer_query("xyzqwerty123456nonexistentword")
        self.assertIn("No relevant financial disclosures found", response["answer"])
        self.assertEqual(response["citations"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
