"""
Political Compass — RAG-ассистент для анализа партийных программ России.

Запуск:
    poetry run streamlit run political_assistant.py
"""

import streamlit as st

from rag_pipeline import PoliticalRAGPipeline

st.set_page_config(
    page_title="Political Compass",
    page_icon="🗳️",
    layout="wide",
)


@st.cache_resource(show_spinner="Загрузка векторного индекса...")
def load_pipeline() -> PoliticalRAGPipeline | None:
    try:
        return PoliticalRAGPipeline()
    except (FileNotFoundError, EnvironmentError) as e:
        st.session_state["pipeline_error"] = str(e)
        return None


def render_setup_banner(error_msg: str) -> None:
    st.error("Ассистент не готов к работе")
    st.markdown(f"```\n{error_msg}\n```")
    st.markdown("""
**Инструкции по настройке:**

1. Скопируйте файл конфигурации:
   ```bash
   cp .env.example .env
   ```

2. Получите бесплатный ключ на [openrouter.ai](https://openrouter.ai) и добавьте в `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

3. Запустите индексацию документов:
   ```bash
   poetry run python ingest.py
   ```

4. Перезапустите приложение.
""")


def render_sources(docs: list) -> None:
    if not docs:
        return
    with st.expander(f"📚 Источники ({len(docs)} фрагментов)"):
        for doc in docs:
            meta = doc.metadata
            party = meta.get("party_display", meta.get("party", ""))
            year = meta.get("year", "")
            chunk_id = meta.get("chunk_id", "")
            st.caption(f"**{party}** | {year} | {chunk_id}")
            st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
            st.divider()


def get_sidebar_config(pipeline: PoliticalRAGPipeline) -> dict:
    with st.sidebar:
        st.title("⚙️ Настройки")

        available_parties = pipeline.get_available_parties()

        st.subheader("Фильтр по партиям")
        if not available_parties:
            st.caption("Партии не найдены в индексе")
            selected_parties = []
        else:
            all_selected = st.checkbox("Все партии", value=True, key="all_parties")
            if all_selected:
                selected_parties = None  # None = без фильтра
                for party in available_parties:
                    st.checkbox(party, value=True, disabled=True, key=f"party_{party}")
            else:
                selected_parties = [
                    party for party in available_parties
                    if st.checkbox(party, value=True, key=f"party_{party}")
                ]
                if not selected_parties:
                    st.warning("Выберите хотя бы одну партию")
                    selected_parties = None

        st.subheader("Режим анализа")
        compare_mode = st.toggle(
            "Сравнить партии",
            value=False,
            help="Отдельно извлекает позиции каждой партии и сравнивает их",
        )

        if compare_mode and (selected_parties is None or len(selected_parties) < 2):
            st.warning("Для сравнения выберите 2+ партии (снимите галочку «Все партии»)")
            compare_mode = False

        st.divider()
        if st.button("🗑️ Очистить чат"):
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Партий в индексе: {len(available_parties)}")

    return {
        "party_filter": selected_parties,
        "compare_mode": compare_mode,
    }


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])


def handle_user_input(prompt: str, pipeline: PoliticalRAGPipeline, config: dict) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую документы..."):
            try:
                answer, docs = pipeline.ask(
                    question=prompt,
                    party_filter=config["party_filter"],
                    compare_mode=config["compare_mode"],
                )
            except Exception as e:
                answer = f"Ошибка при обращении к модели: {e}"
                docs = []

        st.markdown(answer)
        render_sources(docs)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
    })


def main() -> None:
    st.title("🗳️ Political Compass")
    st.caption("RAG-ассистент для анализа программ политических партий России")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    pipeline = load_pipeline()

    if pipeline is None:
        error = st.session_state.get("pipeline_error", "Неизвестная ошибка")
        render_setup_banner(error)
        return

    config = get_sidebar_config(pipeline)
    render_chat_history()

    placeholder_text = (
        "Например: Как КПРФ и ЛДПР относятся к национализации?"
        if config["compare_mode"]
        else "Например: Какова позиция КПРФ по вопросам приватизации?"
    )

    if prompt := st.chat_input(placeholder_text):
        handle_user_input(prompt, pipeline, config)


if __name__ == "__main__":
    main()
