"""
Document Loader & Chunking Module
Extracts textual content from PDF and TXT financial filings, preserves
page/section metadata, and creates overlapping semantic chunks.
"""

from pathlib import Path
from typing import Any
import re
import config


class FinancialDocumentChunk:
    """Represents a single verifiable chunk extracted from a filing."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        page_number: int,
        section: str = "GENERAL",
    ):
        self.chunk_id = chunk_id
        self.text = text.strip()
        self.source = source
        self.page_number = page_number
        self.section = section

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for vector storage and JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page_number": self.page_number,
            "section": self.section,
        }

    def __repr__(self) -> str:
        return f"<Chunk {self.chunk_id} [{self.section}] Page {self.page_number} ({len(self.text)} chars)>"


class FinancialDocumentLoader:
    """Handles parsing and chunking of financial reports (PDF & TXT)."""

    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_chunk_file(self, file_path: str | Path) -> list[FinancialDocumentChunk]:
        """
        Parses a document file and returns structured chunks with page & section attribution.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Filing not found at: {path}")

        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path)
        return self._parse_text(path)

    def _detect_section(self, text: str, current_section: str) -> str:
        """Identifies standard SEC 10-K section headers."""
        patterns = [
            (r"ITEM\s+1A\.?\s+RISK\s+FACTORS", "ITEM 1A - Risk Factors"),
            (r"ITEM\s+1\.?\s+BUSINESS", "ITEM 1 - Business Overview"),
            (r"ITEM\s+7\.?\s+MANAGEMENT", "ITEM 7 - MD&A"),
            (r"ITEM\s+8\.?\s+FINANCIAL\s+STATEMENTS", "ITEM 8 - Financial Statements"),
            (r"PART\s+[I|V|X]+", "SEC Part Header"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return label
        return current_section

    def _parse_text(self, path: Path) -> list[FinancialDocumentChunk]:
        """Parses plain text SEC disclosures."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        # Split text into simulated pages (every ~2500 characters represents 1 SEC filing page)
        simulated_page_size = 2500
        pages: list[tuple[int, str]] = []
        for i in range(0, len(raw_text), simulated_page_size):
            page_num = (i // simulated_page_size) + 1
            pages.append((page_num, raw_text[i : i + simulated_page_size]))

        return self._chunk_pages(pages, source_name=path.name)

    def _parse_pdf(self, path: Path) -> list[FinancialDocumentChunk]:
        """Parses PDF pages using pypdf."""
        pages: list[tuple[int, str]] = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                pages.append((idx + 1, extracted))
        except ImportError:
            # Graceful fallback if pypdf is not yet installed in active environment
            with open(path, "rb") as f:
                raw = f.read().decode("latin1", errors="ignore")
            pages.append((1, raw[:10000]))

        return self._chunk_pages(pages, source_name=path.name)

    def _chunk_pages(self, pages: list[tuple[int, str]], source_name: str) -> list[FinancialDocumentChunk]:
        """Slices pages into overlapping chunks while preserving lineage."""
        chunks: list[FinancialDocumentChunk] = []
        current_section = "GENERAL"
        chunk_counter = 0

        for page_num, text in pages:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            buffer = ""

            for p in paragraphs:
                current_section = self._detect_section(p, current_section)

                if len(buffer) + len(p) <= self.chunk_size:
                    buffer += ("\n" + p if buffer else p)
                else:
                    if buffer:
                        chunk_counter += 1
                        chunk_id = f"{source_name}_p{page_num}_c{chunk_counter}"
                        chunks.append(
                            FinancialDocumentChunk(
                                chunk_id=chunk_id,
                                text=buffer,
                                source=source_name,
                                page_number=page_num,
                                section=current_section,
                            )
                        )
                        # Retain overlap from end of previous buffer
                        buffer = buffer[-self.chunk_overlap :] + "\n" + p
                    else:
                        buffer = p

            if buffer:
                chunk_counter += 1
                chunk_id = f"{source_name}_p{page_num}_c{chunk_counter}"
                chunks.append(
                    FinancialDocumentChunk(
                        chunk_id=chunk_id,
                        text=buffer,
                        source=source_name,
                        page_number=page_num,
                        section=current_section,
                    )
                )

        return chunks
