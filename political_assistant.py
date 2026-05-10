"""
Political Compass - RAG assistant for analyzing Russian political party programs.

Usage:
    poetry run streamlit run political_assistant.py
"""

import streamlit as st

from rag_pipeline import PoliticalRAGPipeline

st.set_page_config(
    page_title="Political Compass",
    page_icon="🗳️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading vector index...")
def load_pipeline() -> PoliticalRAGPipeline | None:
    try:
        return PoliticalRAGPipeline()
    except (FileNotFoundError, EnvironmentError) as e:
        st.session_state["pipeline_error"] = str(e)
        return None


def render_setup_banner(error_msg: str) -> None:
    st.error("The assistant is not ready")
    st.markdown(f"```\n{error_msg}\n```")
    st.markdown("""
**Setup instructions:**

1. Copy the configuration file:
   ```bash
   cp .env.example .env
   ```

2. Get a free key at [openrouter.ai](https://openrouter.ai) and add it to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

3. Index the documents:
   ```bash
   poetry run python ingest.py
   ```

4. Restart the app.
""")


def render_sources(docs: list) -> None:
    if not docs:
        return
    with st.expander(f"📚 Sources ({len(docs)} chunks)"):
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
        st.title("⚙️ Settings")

        available_parties = pipeline.get_available_parties()

        st.subheader("Party filter")
        if not available_parties:
            st.caption("No parties found in the index")
            selected_parties = []
        else:
            all_selected = st.checkbox("All parties", value=True, key="all_parties")
            if all_selected:
                selected_parties = None  # None means no filter.
                for party in available_parties:
                    st.checkbox(party, value=True, disabled=True, key=f"party_{party}")
            else:
                selected_parties = [
                    party for party in available_parties
                    if st.checkbox(party, value=True, key=f"party_{party}")
                ]
                if not selected_parties:
                    st.warning("Select at least one party")
                    selected_parties = None

        st.subheader("Analysis mode")
        compare_mode = st.toggle(
            "Compare parties",
            value=False,
            help="Retrieves each party's position separately and compares them",
        )

        if compare_mode and (selected_parties is None or len(selected_parties) < 2):
            st.warning("Select 2+ parties for comparison (clear the All parties checkbox)")
            compare_mode = False

        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Parties in index: {len(available_parties)}")

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
        with st.spinner("Analyzing documents..."):
            try:
                answer, docs = pipeline.ask(
                    question=prompt,
                    party_filter=config["party_filter"],
                    compare_mode=config["compare_mode"],
                )
            except Exception as e:
                answer = f"Model request failed: {e}"
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
    st.caption("RAG assistant for analyzing Russian political party programs")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    pipeline = load_pipeline()

    if pipeline is None:
        error = st.session_state.get("pipeline_error", "Unknown error")
        render_setup_banner(error)
        return

    config = get_sidebar_config(pipeline)
    render_chat_history()

    placeholder_text = (
        "Example: How do KPRF and LDPR approach nationalization?"
        if config["compare_mode"]
        else "Example: What is KPRF's position on privatization?"
    )

    if prompt := st.chat_input(placeholder_text):
        handle_user_input(prompt, pipeline, config)


if __name__ == "__main__":
    main()
