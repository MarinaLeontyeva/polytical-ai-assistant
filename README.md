# Political Compass: RAG Assistant for Party Program Analysis

A Retrieval-Augmented Generation (RAG) system for analyzing policy documents of Russian political parties. Ask a question, and the system will find relevant excerpts from party programs and generate an analytical answer.

## Architecture

```text
+-----------------------------------------------------+
|                   Streamlit UI                      |
|              political_assistant.py                 |
+-----------------------+-----------------------------+
                        |
                        v
+-----------------------------------------------------+
|              RAG Pipeline (rag_pipeline.py)         |
+--------------+---------------+----------------------+
|  FAISS       |  HuggingFace  |  OpenRouter LLM      |
|  Vector DB   |  Embeddings   |  (free models)       |
+--------------+---------------+----------------------+
                        ^
                        |
+-----------------------------------------------------+
|              Ingestion (ingest.py)                  |
|  PDF/TXT -> Chunks -> Embeddings -> FAISS Index      |
+-----------------------------------------------------+
```

## Quick Start

```bash
# 1. Install dependencies.
poetry install

# 2. Configure environment variables.
cp .env.example .env
# Edit .env and add OPENROUTER_API_KEY.

# 3. Run indexing. The project already includes kprf_2021.pdf.
poetry run python ingest.py

# 4. Start the application.
poetry run streamlit run political_assistant.py
```

## Student Assignment

The base project contains one party program (KPRF). Your task is to extend the system and build a full political compass.

### Step 1. Add Other Party Programs

Add documents to the `data/` directory. File names must follow this format:

```text
{party}_{year}.pdf   or   {party}_{year}.txt
```

Example file names:

```text
ldpr_2021.pdf
edinaya_rossiya_2021.pdf
spravedlivaya_rossiya_2021.pdf
novye_lyudi_2021.pdf
yabloko_2021.pdf
```

Recommended dataset:

| Party | Years | Sources |
|-------|-------|---------|
| KPRF | 2016, 2021 | kprf.ru/party/program |
| LDPR | 2016, 2021 | ldpr.ru/programm |
| United Russia | 2016, 2021 | er.ru/activity/docs |
| A Just Russia | 2016, 2021 | spravedlivo.ru |
| New People | 2021 | newpeople.ru |
| Yabloko | 2016, 2021 | yabloko.ru/program |

After adding documents, rebuild the index:

```bash
poetry run python ingest.py --force-rebuild
```

> **Tip:** You can download PDFs from official party websites. If a PDF is unavailable, copy the program text into a `.txt` file.

### Step 2. Tune Vector Search

Open [ingest.py](ingest.py) and experiment with chunking parameters:

```python
# Current settings around line 60 in ingest.py:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # chunk size in characters
    chunk_overlap=150,   # overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

**Task:** Try different `chunk_size` values (500, 800, 1200) and `chunk_overlap` values (50, 150, 250). How do they affect answer quality? Record your results.

**Extra:** Change the embedding model in `ingest.py` and `rag_pipeline.py`:

```python
# Options, ordered by quality up / speed down:
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # ~120MB, fast
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"    # ~280MB
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"   # ~560MB, best quality
```

### Step 3. Improve Prompts

Open [rag_pipeline.py](rag_pipeline.py). Find the `SINGLE_SYSTEM_PROMPT` and `COMPARE_SYSTEM_PROMPT` variables.

**Task:** Modify the prompts for different scenarios:

1. **Basic analysis** - the current prompt for one party.
2. **Comparative analysis** - the current prompt for comparison.
3. **Quote search** - a prompt that returns exact quotes from party programs.
4. **Ideological classification** - a prompt that places a party on a political compass (left/right, liberal/conservative).

Example prompt rule change:

```python
SINGLE_SYSTEM_PROMPT = """You are a neutral political analyst...

Answer rules:
1. [your rules]
...
```

**Hint:** A good political analysis prompt should:

- Prevent hallucinations by requiring references to the text.
- Define the answer structure.
- Explicitly handle missing information.
- Ensure a neutral tone.

### Step 4. Improve the Interface

Open [political_assistant.py](political_assistant.py) and add:

1. **Source counter chips:**

   ```python
   # Show parties found in the sources as colored badges.
   for party in set(doc.metadata.get("party_display") for doc in docs):
       st.badge(party)
   ```

2. **Example questions** with quick buttons for common queries:

   ```python
   example_questions = [
       "What do parties say about the retirement age?",
       "How do parties approach small business?",
       "What is each party's position on foreign policy?",
   ]
   for q in example_questions:
       if st.button(q):
           handle_user_input(q, pipeline, config)
   ```

3. **Year filter** to choose programs from 2016 or 2021.

4. **Answer export** with a button to copy the answer and sources.

## Test Questions

After adding several parties, test the system:

| Category | Question |
|----------|----------|
| Economy | What are the parties' positions on industrial nationalization? |
| Social policy | What support measures for pensioners do the parties propose? |
| Foreign policy | How do the parties assess Russia's relations with the West? |
| Ideology | Which parties use the term "patriotism", and how do they interpret it? |
| Missing data | What do the parties think about cryptocurrencies? Expected answer: "not found". |

## Project Structure

```text
.
+-- political_assistant.py  # Streamlit UI
+-- rag_pipeline.py         # RAG: search and answer generation
+-- ingest.py               # Document indexing
+-- pyproject.toml          # Poetry dependencies
+-- .env.example            # Environment variable template
+-- data/
    +-- kprf_2021.pdf       # Example document (KPRF)
```

## Grading Criteria

| Criterion | Weight |
|-----------|--------|
| Data loading and processing (3+ parties, metadata) | 15% |
| Vector search tuning (parameter experiments) | 15% |
| RAG answer quality (relevance, no hallucinations) | 25% |
| Prompt quality (neutrality, structure, rules) | 20% |
| Interface improvements | 15% |
| Presentation and architecture explanation | 10% |

## Useful Links

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [OpenRouter free models](https://openrouter.ai/models?q=free)
- [Streamlit Chat Documentation](https://docs.streamlit.io/develop/api-reference/chat)
- [FAISS Vector Store (LangChain)](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
- [Multilingual E5 Embeddings](https://huggingface.co/intfloat/multilingual-e5-small)
