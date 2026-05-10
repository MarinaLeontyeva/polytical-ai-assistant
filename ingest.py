"""
Script for indexing political party programs into a FAISS vector database.

Usage:
    python ingest.py
    python ingest.py --force-rebuild   # rebuild the index

File naming format in the data/ directory:
    {party}_{year}.pdf  or  {party}_{year}.txt
    Examples: kprf_2021.pdf, ldpr_2016.txt, edinaya_rossiya_2021.pdf
"""

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROGRAMS_DIR = os.getenv("PROGRAMS_DIR", "./data")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # ~120MB, works well with Russian text

PARTY_DISPLAY_NAMES = {
    "kprf": "КПРФ",
    "ldpr": "ЛДПР",
    "edinaya_rossiya": "Единая Россия",
    "edinaya": "Единая Россия",
    "spravedlivaya_rossiya": "Справедливая Россия",
    "spravedlivaya": "Справедливая Россия",
    "novye_lyudi": "Новые люди",
    "novye": "Новые люди",
    "yabloko": "Яблоко",
}


def get_display_name(slug: str) -> str:
    return PARTY_DISPLAY_NAMES.get(slug.lower(), slug.upper())


def parse_filename_metadata(filepath: str) -> dict:
    """Extract party and year metadata from a file name."""
    name = Path(filepath).stem  # "kprf_2021"
    match = re.match(r"^(.+?)_(\d{4})$", name)
    if not match:
        raise ValueError(
            f"File '{Path(filepath).name}' does not match the {{party}}_{{year}} format.\n"
            f"Rename the file, for example: kprf_2021.pdf"
        )
    party_slug = match.group(1).lower()
    year = match.group(2)
    return {
        "party": party_slug,
        "party_display": get_display_name(party_slug),
        "year": year,
        "source": Path(filepath).name,
    }


def load_document(filepath: str) -> list:
    """Load a PDF or TXT file as a list of Document objects."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return loader.load()


def load_all_documents(programs_dir: str) -> list:
    """Load all documents from a directory and add metadata from file names."""
    documents = []
    skipped = []
    programs_path = Path(programs_dir)

    if not programs_path.exists():
        print(f"Directory '{programs_dir}' was not found.")
        sys.exit(1)

    files = sorted(programs_path.glob("*.pdf")) + sorted(programs_path.glob("*.txt"))
    if not files:
        print(f"Directory '{programs_dir}' contains no PDF or TXT files.")
        sys.exit(1)

    for filepath in files:
        try:
            metadata = parse_filename_metadata(str(filepath))
            docs = load_document(str(filepath))
            for doc in docs:
                doc.metadata.update(metadata)
            documents.extend(docs)
            print(f"  OK {filepath.name} - {len(docs)} pages ({metadata['party_display']}, {metadata['year']})")
        except ValueError as e:
            skipped.append(str(e))
            print(f"  Skipped: {e}")

    if not documents:
        print("No valid documents were found.")
        sys.exit(1)

    return documents


def chunk_documents(docs: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Add chunk_id values so sources can be traced back.
    party_counters: dict[str, int] = {}
    for chunk in chunks:
        party = chunk.metadata.get("party", "unknown")
        year = chunk.metadata.get("year", "0000")
        key = f"{party}_{year}"
        party_counters[key] = party_counters.get(key, 0) + 1
        chunk.metadata["chunk_id"] = f"{key}_{party_counters[key]:04d}"

    return chunks


def build_embedder() -> HuggingFaceEmbeddings:
    """Load the embedding model for Russian-language text."""
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    print("(On the first run, the model downloads ~120MB and may take a few minutes.)")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_and_save_faiss(chunks: list, embedder: HuggingFaceEmbeddings, index_path: str) -> FAISS:
    """Build a FAISS index and save it to disk."""
    print(f"Creating a vector index for {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(index_path)
    print(f"Index saved: {index_path}/")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Index political party programs")
    parser.add_argument("--programs-dir", default=PROGRAMS_DIR, help="Directory with documents")
    parser.add_argument("--index-path", default=FAISS_INDEX_PATH, help="Path for saving the index")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the index")
    args = parser.parse_args()

    index_path = args.index_path
    programs_dir = args.programs_dir

    if Path(index_path).exists() and not args.force_rebuild:
        print(f"Index already exists: {index_path}/")
        print("Use --force-rebuild to rebuild it.")
        return

    print(f"Loading documents from '{programs_dir}'...")
    docs = load_all_documents(programs_dir)
    print(f"Loaded pages: {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"Created chunks: {len(chunks)}")

    embedder = build_embedder()
    build_and_save_faiss(chunks, embedder, index_path)

    print("\nIndexing complete. Start the app:")
    print("  poetry run streamlit run political_assistant.py")


if __name__ == "__main__":
    main()
