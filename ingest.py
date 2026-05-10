"""
Скрипт для индексации партийных программ в векторную базу данных FAISS.

Запуск:
    python ingest.py
    python ingest.py --force-rebuild   # пересоздать индекс

Формат файлов в папке data/:
    {партия}_{год}.pdf  или  {партия}_{год}.txt
    Примеры: kprf_2021.pdf, ldpr_2016.txt, edinaya_rossiya_2021.pdf
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
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # ~120MB, хорошо работает с русским

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
    """Извлекает метаданные партии и года из имени файла."""
    name = Path(filepath).stem  # "kprf_2021"
    match = re.match(r"^(.+?)_(\d{4})$", name)
    if not match:
        raise ValueError(
            f"Файл '{Path(filepath).name}' не соответствует формату {{партия}}_{{год}}.\n"
            f"Переименуйте файл, например: kprf_2021.pdf"
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
    """Загружает PDF или TXT файл как список Document объектов."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")
    return loader.load()


def load_all_documents(programs_dir: str) -> list:
    """Загружает все документы из папки, добавляет метаданные из имён файлов."""
    documents = []
    skipped = []
    programs_path = Path(programs_dir)

    if not programs_path.exists():
        print(f"Папка '{programs_dir}' не найдена.")
        sys.exit(1)

    files = sorted(programs_path.glob("*.pdf")) + sorted(programs_path.glob("*.txt"))
    if not files:
        print(f"В папке '{programs_dir}' нет файлов PDF или TXT.")
        sys.exit(1)

    for filepath in files:
        try:
            metadata = parse_filename_metadata(str(filepath))
            docs = load_document(str(filepath))
            for doc in docs:
                doc.metadata.update(metadata)
            documents.extend(docs)
            print(f"  ✓ {filepath.name} — {len(docs)} стр. ({metadata['party_display']}, {metadata['year']})")
        except ValueError as e:
            skipped.append(str(e))
            print(f"  ✗ Пропущен: {e}")

    if not documents:
        print("Не найдено ни одного корректного документа.")
        sys.exit(1)

    return documents


def chunk_documents(docs: list) -> list:
    """Разбивает документы на фрагменты с перекрытием."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Добавляем chunk_id для отслеживаемости источников
    party_counters: dict[str, int] = {}
    for chunk in chunks:
        party = chunk.metadata.get("party", "unknown")
        year = chunk.metadata.get("year", "0000")
        key = f"{party}_{year}"
        party_counters[key] = party_counters.get(key, 0) + 1
        chunk.metadata["chunk_id"] = f"{key}_{party_counters[key]:04d}"

    return chunks


def build_embedder() -> HuggingFaceEmbeddings:
    """Загружает модель эмбеддингов для русского языка."""
    print(f"Загрузка модели эмбеддингов {EMBEDDING_MODEL}...")
    print("(При первом запуске модель скачивается ~120MB, это займёт пару минут)")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_and_save_faiss(chunks: list, embedder: HuggingFaceEmbeddings, index_path: str) -> FAISS:
    """Строит FAISS индекс и сохраняет на диск."""
    print(f"Создание векторного индекса для {len(chunks)} фрагментов...")
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(index_path)
    print(f"Индекс сохранён: {index_path}/")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Индексация партийных программ")
    parser.add_argument("--programs-dir", default=PROGRAMS_DIR, help="Папка с документами")
    parser.add_argument("--index-path", default=FAISS_INDEX_PATH, help="Путь для сохранения индекса")
    parser.add_argument("--force-rebuild", action="store_true", help="Пересоздать индекс")
    args = parser.parse_args()

    index_path = args.index_path
    programs_dir = args.programs_dir

    if Path(index_path).exists() and not args.force_rebuild:
        print(f"Индекс уже существует: {index_path}/")
        print("Используйте --force-rebuild для пересоздания.")
        return

    print(f"Загрузка документов из '{programs_dir}'...")
    docs = load_all_documents(programs_dir)
    print(f"Загружено страниц: {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"Создано фрагментов: {len(chunks)}")

    embedder = build_embedder()
    build_and_save_faiss(chunks, embedder, index_path)

    print("\nИндексация завершена! Запустите приложение:")
    print("  poetry run streamlit run political_assistant.py")


if __name__ == "__main__":
    main()
