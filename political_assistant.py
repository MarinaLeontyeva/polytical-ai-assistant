"""
Political Compass - RAG assistant for analyzing Russian political party programs.

Supports 4 analysis modes:
  - Single:    answer about a specific party
  - Compare:   side-by-side comparison of several parties
  - Quotes:    direct verbatim quotes from party programs on a topic
  - Ideology:  ideological classification of parties on 3 axes

Usage:
    streamlit run political_assistant.py
"""
from datetime import datetime

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

from rag_pipeline import PoliticalRAGPipeline, format_context


# ============================================================
# EXTRA MODES (Quotes and Ideology) — attached to the pipeline
# ============================================================
QUOTES_PROMPT = """You are a research assistant. Your task is to find and return EXACT quotes from political party programs that address the user's topic.

Rules:
1. Return ONLY verbatim quotes from the provided excerpts. Do not paraphrase.
2. Each quote should be 1-3 sentences. Pick the most relevant portion if a passage is long.
3. Format each quote as:
   > "exact quote text"
   — [Party, Year, chunk XXXX]
4. Group quotes by party.
5. If no relevant quotes exist, write: "No direct quotes found in the analyzed documents."
6. Do NOT add commentary, summary, or analysis between quotes.

Document excerpts:
{context}"""

IDEOLOGY_PROMPT = """You are a political science analyst. Based on the provided excerpts, place each listed party on three ideological axes:

Axes:
1. **Economic**: state-controlled (left) ↔ market-based (right)
2. **Social values**: progressive ↔ conservative
3. **State role**: strong state intervention ↔ minimal state

For EACH party, output:
**Party Name**
- Economic: [position] — justification with [Party, Year, chunk XXXX]
- Social values: [position] — justification with markers
- State role: [position] — justification with markers

End with a brief comparison table.

Strict rules:
- Use ONLY evidence from the provided excerpts.
- If a party's position on an axis is not in the documents, write "insufficient data" for that axis.
- Use neutral academic English.

Document excerpts:
{context}"""


def _ask_quotes(self, topic, party_filter=None):
    docs = self.retrieve(topic, party_filter, k=8)
    if not docs:
        return "No relevant excerpts found.", []
    context = format_context(docs)
    messages = [
        SystemMessage(content=QUOTES_PROMPT.format(context=context)),
        HumanMessage(content=topic),
    ]
    return self._llm.invoke(messages).content, docs


def _ask_ideology(self, parties):
    all_docs = []
    for party in parties:
        party_docs = self.retrieve(
            "economic policy state role social values ideology",
            party_filter=[party], k=5
        )
        all_docs.extend(party_docs)
    if not all_docs:
        return "No relevant excerpts found.", []
    context = format_context(all_docs)
    messages = [
        SystemMessage(content=IDEOLOGY_PROMPT.format(context=context)),
        HumanMessage(content=f"Classify these parties: {', '.join(parties)}"),
    ]
    return self._llm.invoke(messages).content, all_docs


PoliticalRAGPipeline.ask_quotes = _ask_quotes
PoliticalRAGPipeline.ask_ideology = _ask_ideology


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Political Compass",
    page_icon="🗳️",
    layout="wide",
)


# ============================================================
# EXAMPLE QUESTIONS (one per mode, from lecture slide 22)
# ============================================================
EXAMPLE_QUESTIONS = {
    "Single": "What is United Russia's position on pension reform?",
    "Compare": "Compare the economic policies of CPRF and LDPR.",
    "Quotes": "Find direct quotes about social spending and welfare.",
    "Ideology": "Classify the selected parties ideologically.",
}


# ============================================================
# PIPELINE LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading vector index...")
def load_pipeline() -> PoliticalRAGPipeline | None:
    # Auto-build FAISS index on first startup (Streamlit Cloud starts without one)
    from pathlib import Path
    import subprocess
    if not Path("faiss_index").exists():
        with st.spinner("Building vector index for the first time (5-7 minutes)..."):
            result = subprocess.run(
                ["python", "ingest.py"],
                capture_output=True, text=True, timeout=900,
            )
            if result.returncode != 0:
                st.session_state["pipeline_error"] = (
                    f"Failed to build index:\n{result.stderr[-1500:]}"
                )
                return None
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
   python ingest.py
   ```
4. Restart the app.
""")


# ============================================================
# SOURCES RENDERING (with party badges)
# ============================================================
PARTY_COLORS = {
    "КПРФ": "#D32F2F",
    "ЛДПР": "#FFD600",
    "Единая Россия": "#1976D2",
    "Справедливая Россия": "#E91E63",
    "Новые люди": "#43A047",
    "Яблоко": "#7B1FA2",
}


def party_badge(party: str) -> str:
    color = PARTY_COLORS.get(party, "#607D8B")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600">{party}</span>'
    )


def render_sources(docs: list) -> None:
    if not docs:
        return
    with st.expander(f"📚 Sources ({len(docs)} chunks)"):
        for doc in docs:
            meta = doc.metadata
            party = meta.get("party_display", meta.get("party", "Unknown"))
            year = meta.get("year", "")
            chunk_id = meta.get("chunk_id", "")
            st.markdown(
                f"{party_badge(party)} &nbsp; <code>{year}</code> &nbsp; "
                f"<small><code>{chunk_id}</code></small>",
                unsafe_allow_html=True,
            )
            preview = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            st.text(preview)
            st.divider()


# ============================================================
# SIDEBAR CONFIG
# ============================================================
def get_sidebar_config(pipeline: PoliticalRAGPipeline) -> dict:
    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("Analysis mode")
        mode = st.radio(
            "Select mode",
            options=["Single", "Compare", "Quotes", "Ideology"],
            help=(
                "Single — analyze one party | "
                "Compare — side-by-side | "
                "Quotes — verbatim quotes | "
                "Ideology — left/right classification"
            ),
            label_visibility="collapsed",
        )

        st.divider()

        available_parties = pipeline.get_available_parties()
        st.subheader("Party filter")
        if not available_parties:
            st.caption("No parties found in the index")
            selected_parties = []
        else:
            all_selected = st.checkbox("All parties", value=True, key="all_parties")
            if all_selected:
                selected_parties = None  # None means no filter
                for party in available_parties:
                    st.checkbox(party, value=True, disabled=True, key=f"party_disabled_{party}")
            else:
                selected_parties = [
                    party for party in available_parties
                    if st.checkbox(party, value=True, key=f"party_{party}")
                ]
                if not selected_parties:
                    st.warning("Select at least one party")
                    selected_parties = None

        # Mode-specific validations
        if mode == "Compare" and (selected_parties is None or len(selected_parties) < 2):
            if selected_parties is None:
                pass  # All parties — fine
            else:
                st.warning("Compare mode needs 2+ parties")

        if mode == "Ideology" and selected_parties is None:
            selected_parties = available_parties  # Use all for Ideology

        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Parties in index: {len(available_parties)}")

    return {
        "mode": mode,
        "party_filter": selected_parties,
    }


# ============================================================
# EXAMPLE QUESTION BUTTONS
# ============================================================
def render_example_buttons(current_mode: str) -> str | None:
    """Show 1-click buttons with example questions. Returns the clicked question, if any."""
    if st.session_state.get("messages"):
        return None  # Hide examples once the conversation has started

    st.markdown("**💡 Try an example:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, (mode_label, question) in zip(cols, EXAMPLE_QUESTIONS.items()):
        with col:
            highlight = " (current mode)" if mode_label == current_mode else ""
            if st.button(f"**{mode_label}**{highlight}\n\n{question}", key=f"ex_{mode_label}"):
                return question
    return None


# ============================================================
# CHAT HISTORY
# ============================================================
def render_chat_history() -> None:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            # Export button for assistant messages
            if message["role"] == "assistant":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="⬇️ Download answer (Markdown)",
                    data=message["content"],
                    file_name=f"political_compass_answer_{ts}.md",
                    mime="text/markdown",
                    key=f"download_{i}",
                )


# ============================================================
# DISPATCH TO PIPELINE BY MODE
# ============================================================
def run_pipeline(prompt: str, pipeline: PoliticalRAGPipeline, config: dict):
    mode = config["mode"]
    party_filter = config["party_filter"]

    if mode == "Single":
        return pipeline.ask(prompt, party_filter=party_filter, compare_mode=False)
    if mode == "Compare":
        return pipeline.ask(prompt, party_filter=party_filter, compare_mode=True)
    if mode == "Quotes":
        return pipeline.ask_quotes(prompt, party_filter=party_filter)
    if mode == "Ideology":
        # Ideology uses parties list, prompt content is ignored
        parties = party_filter if party_filter else pipeline.get_available_parties()
        return pipeline.ask_ideology(parties)

    raise ValueError(f"Unknown mode: {mode}")


def handle_user_input(prompt: str, pipeline: PoliticalRAGPipeline, config: dict) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Running {config['mode']} analysis..."):
            try:
                answer, docs = run_pipeline(prompt, pipeline, config)
            except Exception as e:
                answer = f"Model request failed: {e}"
                docs = []

        st.markdown(answer)
        render_sources(docs)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download answer (Markdown)",
            data=answer,
            file_name=f"political_compass_answer_{ts}.md",
            mime="text/markdown",
            key=f"download_latest_{ts}",
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
    })


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    st.title("🗳️ Political Compass")
    st.caption("RAG assistant for analyzing Russian political party programs (2021)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    pipeline = load_pipeline()

    if pipeline is None:
        error = st.session_state.get("pipeline_error", "Unknown error")
        render_setup_banner(error)
        return

    config = get_sidebar_config(pipeline)

    # Example question buttons (shown only at the start of conversation)
    example_clicked = render_example_buttons(config["mode"])

    render_chat_history()

    # Mode-specific input placeholder
    placeholders = {
        "Single": "Ask about a specific party (e.g., 'What is KPRF position on healthcare?')",
        "Compare": "Ask a comparative question (e.g., 'Compare tax policies')",
        "Quotes": "Topic to find direct quotes about (e.g., 'education')",
        "Ideology": "Press Enter or type anything — uses selected parties",
    }
    placeholder = placeholders.get(config["mode"], "Ask a question...")

    # Either example button click OR chat input triggers the query
    user_prompt = example_clicked or st.chat_input(placeholder)

    if user_prompt:
        handle_user_input(user_prompt, pipeline, config)
        st.rerun()


if __name__ == "__main__":
    main()
