"""
PDF Parser - Extracts and chunks text from PDFs in the documents/ folder.
"""

import pdfplumber
import io
import os
from typing import List


def extract_text_from_pdf_file(file_path: str) -> str:
    """Extract all text from a PDF file path."""
    text_pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""
    return "\n\n".join(text_pages)


def load_all_documents(documents_folder: str = "documents") -> tuple:
    """
    Load all PDFs from the documents folder.

    Returns:
        (all_chunks, doc_names) tuple
    """
    all_chunks = []
    doc_names = []

    if not os.path.exists(documents_folder):
        return [], []

    pdf_files = [f for f in os.listdir(documents_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        return [], []

    for filename in pdf_files:
        file_path = os.path.join(documents_folder, filename)
        text = extract_text_from_pdf_file(file_path)
        if text.strip():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            doc_names.append(filename)
            print(f"  ✅ Loaded: {filename} ({len(chunks)} chunks)")

    return all_chunks, doc_names


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for better RAG retrieval."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end < text_length:
            for boundary in ['. ', '.\n', '\n\n', '! ', '? ']:
                boundary_pos = text.rfind(boundary, start, end)
                if boundary_pos != -1 and boundary_pos > start + chunk_size // 2:
                    end = boundary_pos + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks
