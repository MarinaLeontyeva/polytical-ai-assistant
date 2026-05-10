"""
RAG pipeline for analyzing political party programs.

Uses:
- FAISS for vector search
- OpenRouter (free models) for answer generation
- HuggingFace Embeddings for Russian-language embeddings
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SINGLE_SYSTEM_PROMPT = """You are a neutral political analyst. Your task is to objectively analyze policy documents of Russian political parties.

Answer rules:
1. Answer strictly based on the provided document excerpts.
2. Do not express a personal opinion or judge positions as "right" or "wrong".
3. Use neutral academic Russian.
4. If there is not enough information, clearly state: "This information was not found in the analyzed documents."
5. Include the party and source year when referring to a specific position.
6. Structure the answer with a direct answer first, then arguments from the program.

Political party document excerpts:
{context}"""

COMPARE_SYSTEM_PROMPT = """You are a neutral political analyst. Compare the specified parties' positions on the given question based on their policy documents.

Answer structure:
1. **Brief introduction** (1-2 sentences about the topic)
2. **Each party's position** (a separate section for each party)
3. **Key similarities and differences**
4. **Neutral summary**

Rules:
- Use only facts from the documents, with no political judgments.
- If a party's position on the question is not reflected in the documents, state this clearly.

Document excerpts by party:
{context}"""


def format_context(docs: list) -> str:
    """Format a list of documents into text context with headers."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        party = meta.get("party_display", meta.get("party", "Unknown party"))
        year = meta.get("year", "")
        chunk_id = meta.get("chunk_id", "")
        header = f"[{party} | {year} | chunk {chunk_id}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class PoliticalRAGPipeline:
    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run indexing first:\n"
                "  python ingest.py"
            )
        if not OPENROUTER_API_KEY:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set.\n"
                "Get a free key at https://openrouter.ai and add it to .env"
            )

        self._embedder = self._load_embedder()
        self._vectorstore = self._load_vectorstore(index_path)
        self._llm = self._load_llm()

    def _load_embedder(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_vectorstore(self, index_path: str) -> FAISS:
        return FAISS.load_local(
            index_path,
            self._embedder,
            allow_dangerous_deserialization=True,
        )

    def _load_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/political-compass-rag",
                "X-Title": "Political Compass RAG",
            },
        )

    def get_available_parties(self) -> list[str]:
        """Return the list of parties available in the index."""
        parties = set()
        for doc in self._vectorstore.docstore._dict.values():
            display = doc.metadata.get("party_display") or doc.metadata.get("party", "")
            if display:
                parties.add(display)
        return sorted(parties)

    def retrieve(self, query: str, party_filter: list[str] | None = None, k: int = 5) -> list:
        """Find relevant chunks, optionally filtering by party."""
        if party_filter:
            # Retrieve extra candidates for filtering.
            candidates = self._vectorstore.similarity_search(query, k=k * 6)
            filtered = [
                doc for doc in candidates
                if doc.metadata.get("party_display") in party_filter
                or doc.metadata.get("party") in party_filter
            ]
            return filtered[:k]
        return self._vectorstore.similarity_search(query, k=k)

    def ask(
        self,
        question: str,
        party_filter: list[str] | None = None,
        compare_mode: bool = False,
    ) -> tuple[str, list]:
        """
        Ask a question and return (answer, source documents).

        Args:
            question: user question
            party_filter: list of parties for filtering (None = all parties)
            compare_mode: party comparison mode (uses a different prompt)
        """
        if compare_mode and party_filter and len(party_filter) >= 2:
            return self._ask_compare(question, party_filter)
        return self._ask_single(question, party_filter)

    def _ask_single(self, question: str, party_filter: list[str] | None) -> tuple[str, list]:
        docs = self.retrieve(question, party_filter, k=5)
        if not docs:
            return "This information was not found in the analyzed documents.", []

        context = format_context(docs)
        messages = [
            SystemMessage(content=SINGLE_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]
        response = self._llm.invoke(messages)
        return response.content, docs

    def _ask_compare(self, question: str, parties: list[str]) -> tuple[str, list]:
        all_docs = []
        for party in parties:
            party_docs = self.retrieve(question, party_filter=[party], k=3)
            all_docs.extend(party_docs)

        if not all_docs:
            return "This information was not found in the analyzed documents.", []

        context = format_context(all_docs)
        messages = [
            SystemMessage(content=COMPARE_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]
        response = self._llm.invoke(messages)
        return response.content, all_docs
