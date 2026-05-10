"""
RAG-пайплайн для анализа партийных программ.

Использует:
- FAISS для векторного поиска
- OpenRouter (бесплатные модели) для генерации ответов
- HuggingFace Embeddings для русскоязычных эмбеддингов
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

SINGLE_SYSTEM_PROMPT = """Ты — нейтральный политический аналитик. Твоя задача — объективно анализировать программные документы политических партий России.

Правила ответа:
1. Отвечай строго на основе предоставленных фрагментов документов.
2. Не выражай личного мнения и не оценивай позиции как «правильные» или «неправильные».
3. Используй нейтральный академический язык на русском.
4. Если информации недостаточно, явно укажи: «В проанализированных документах эта информация не найдена.»
5. Указывай партию и год источника при ссылке на конкретную позицию.
6. Структурируй ответ: сначала прямой ответ, затем аргументы из программы.

Фрагменты партийных документов:
{context}"""

COMPARE_SYSTEM_PROMPT = """Ты — нейтральный политический аналитик. Сравни позиции указанных партий по заданному вопросу на основе их программных документов.

Структура ответа:
1. **Краткое введение** (1–2 предложения о теме)
2. **Позиция каждой партии** (отдельный раздел для каждой)
3. **Ключевые сходства и различия**
4. **Нейтральное резюме**

Правила:
- Только факты из документов, без политических оценок.
- Если позиция партии по вопросу не отражена в документах — укажи это явно.

Фрагменты документов по партиям:
{context}"""


def format_context(docs: list) -> str:
    """Форматирует список документов в текстовый контекст с заголовками."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        party = meta.get("party_display", meta.get("party", "Неизвестная партия"))
        year = meta.get("year", "")
        chunk_id = meta.get("chunk_id", "")
        header = f"[{party} | {year} | фрагмент {chunk_id}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class PoliticalRAGPipeline:
    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"Индекс не найден: {index_path}\n"
                "Сначала запустите индексацию:\n"
                "  python ingest.py"
            )
        if not OPENROUTER_API_KEY:
            raise EnvironmentError(
                "Не задан OPENROUTER_API_KEY.\n"
                "Получите бесплатный ключ на https://openrouter.ai и добавьте его в .env"
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
        """Возвращает список партий, доступных в индексе."""
        parties = set()
        for doc in self._vectorstore.docstore._dict.values():
            display = doc.metadata.get("party_display") or doc.metadata.get("party", "")
            if display:
                parties.add(display)
        return sorted(parties)

    def retrieve(self, query: str, party_filter: list[str] | None = None, k: int = 5) -> list:
        """Находит релевантные фрагменты, опционально фильтруя по партиям."""
        if party_filter:
            # Берём больше кандидатов для последующей фильтрации
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
        Задаёт вопрос и возвращает (ответ, список документов-источников).

        Args:
            question: вопрос пользователя
            party_filter: список партий для фильтрации (None = все партии)
            compare_mode: режим сравнения партий (использует другой промпт)
        """
        if compare_mode and party_filter and len(party_filter) >= 2:
            return self._ask_compare(question, party_filter)
        return self._ask_single(question, party_filter)

    def _ask_single(self, question: str, party_filter: list[str] | None) -> tuple[str, list]:
        docs = self.retrieve(question, party_filter, k=5)
        if not docs:
            return "В проанализированных документах информация по данному вопросу не найдена.", []

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
            return "В проанализированных документах информация по данному вопросу не найдена.", []

        context = format_context(all_docs)
        messages = [
            SystemMessage(content=COMPARE_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]
        response = self._llm.invoke(messages)
        return response.content, all_docs
