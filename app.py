import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

import streamlit as st
import tempfile, os, psutil, time, socket
from dotenv import load_dotenv
from rag.ingestor import ingest_files, load_index, index_exists
from rag.retriever import retrieve
from rag.generator import build_prompt, MODEL, MAX_RETRIES
import rag.generator as gen_module

load_dotenv()

st.set_page_config(page_title="Ask My PDF Bot", page_icon="📄", layout="wide")

# ── Session state ───────────────────────────────────────────────
if "history"      not in st.session_state: st.session_state.history      = []
if "faiss_index"  not in st.session_state: st.session_state.faiss_index  = None
if "bm25"         not in st.session_state: st.session_state.bm25         = None
if "chunks"       not in st.session_state: st.session_state.chunks       = None
if "indexed_docs" not in st.session_state: st.session_state.indexed_docs = []


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Ask My PDF Bot")
    st.caption("RAG · Hybrid Retrieval · Multi-LLM")

    uploaded = st.file_uploader(
        "Upload PDFs or Word docs",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded and st.button("🔍 Index documents", type="primary"):
        progress = st.progress(0, text="Starting…")
        tmp_paths = []

        for i, f in enumerate(uploaded):
            progress.progress(
                int((i / len(uploaded)) * 40),
                text=f"Reading {f.name}…"
            )
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(f.name)[1],
                prefix=os.path.splitext(f.name)[0] + "_"
            )
            tmp.write(f.read())
            tmp.flush()
            tmp_paths.append(tmp.name)

        progress.progress(50, text="Chunking and embedding… (1-2 min for large PDFs)")
        n = ingest_files(tmp_paths)

        progress.progress(90, text="Loading index…")
        st.session_state.faiss_index, \
        st.session_state.bm25,        \
        st.session_state.chunks       = load_index()
        st.session_state.indexed_docs = [f.name for f in uploaded]
        st.session_state.history      = []

        progress.progress(100, text="Done!")
        time.sleep(0.5)
        progress.empty()

        st.success(f"✅ {n} chunks indexed from {len(uploaded)} doc(s)")
        for name in st.session_state.indexed_docs:
            st.caption(f"📄 {name}")

    # Load existing index if present
    if st.session_state.faiss_index is None and index_exists():
        st.session_state.faiss_index, \
        st.session_state.bm25,        \
        st.session_state.chunks       = load_index()
        st.info("Loaded existing index.")

    st.divider()

    # LLM selector
    st.subheader("🤖 LLM Backend")
    llm_choice = st.radio(
        "Choose model",
        ["Groq (fast, free)", "Gemini (API)", "Ollama (offline)"],
        index=0,
        key="llm_choice"
    )

    if llm_choice == "Groq (fast, free)":
        gen_module.USE_GROQ   = True
        gen_module.USE_OLLAMA = False
        st.caption("⚡ Llama3 on Groq — fastest free option")
    elif llm_choice == "Ollama (offline)":
        gen_module.USE_OLLAMA = True
        gen_module.USE_GROQ   = False
        st.caption("🔌 Local Llama 3.2 — no internet needed")
    else:
        gen_module.USE_GROQ   = False
        gen_module.USE_OLLAMA = False
        st.caption("☁️ Google Gemini API")

    st.divider()

    # System monitor
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    st.metric("CPU", f"{cpu:.0f}%")
    st.metric("RAM", f"{ram:.0f}%")

    if st.session_state.history:
        if st.button("🗑 Clear chat"):
            st.session_state.history = []
            st.rerun()


# ── Main chat area ──────────────────────────────────────────────
st.header("Ask your documents")

if st.session_state.faiss_index is None:
    st.info("⬅  Upload and index at least one document to begin.")
    st.stop()

# Render previous history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            with st.expander("📎 Sources"):
                for s in turn["sources"]:
                    st.caption(f"**{s['doc_name']}** — page {s['page']}")
                    st.text(
                        s["text"][:300] + "…"
                        if len(s["text"]) > 300
                        else s["text"]
                    )

# ── Chat input ──────────────────────────────────────────────────
query = st.chat_input("Ask a question about your documents…")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):

        # Step 1 — Retrieve chunks
        with st.spinner("Searching documents…"):
            chunks = retrieve(
                query,
                st.session_state.faiss_index,
                st.session_state.bm25,
                st.session_state.chunks
            )

        # Step 2 — Build prompt
        prompt = build_prompt(query, chunks, st.session_state.history)

        # Step 3 — Generate response
        answer_placeholder = st.empty()
        answer             = ""

        # ── Ollama (offline / local only) ──
        if gen_module.USE_OLLAMA:
            try:
                socket.create_connection(("localhost", 11434), timeout=2)
                with st.spinner("Generating with local Llama…"):
                    answer = gen_module.generate_answer_ollama(prompt)
                answer_placeholder.markdown(answer)
            except OSError:
                answer = (
                    "⚠️ Ollama is not running on this machine. "
                    "Please start Ollama locally or switch to "
                    "Groq / Gemini in the sidebar."
                )
                answer_placeholder.warning(answer)

        # ── Groq (fast, free) ──
        elif gen_module.USE_GROQ:
            try:
                with st.spinner("Generating with Groq…"):
                    answer = gen_module.generate_answer_groq(prompt)
                answer_placeholder.markdown(answer)
            except Exception as e:
                answer = f"⚠️ Groq error: {str(e)}"
                answer_placeholder.error(answer)

        # ── Gemini (streaming) ──
        else:
            import google.generativeai as genai
            from google.api_core.exceptions import ResourceExhausted
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            gemini_model = genai.GenerativeModel(MODEL)
            full_answer  = ""

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = gemini_model.generate_content(prompt, stream=True)
                    for chunk in response:
                        if chunk.text:
                            full_answer += chunk.text
                            answer_placeholder.markdown(full_answer + "▌")
                            time.sleep(0.01)

                    answer_placeholder.markdown(full_answer)
                    answer = full_answer
                    break

                except ResourceExhausted as e:
                    wait = 60
                    try:
                        wait = int(str(e).split("seconds:")[1].split("}")[0].strip()) + 5
                    except Exception:
                        pass

                    if attempt < MAX_RETRIES:
                        answer_placeholder.warning(
                            f"⏳ Rate limited. Retrying in {wait}s "
                            f"(attempt {attempt}/{MAX_RETRIES})…"
                        )
                        time.sleep(wait)
                        answer_placeholder.empty()
                    else:
                        answer = (
                            f"⚠️ Gemini rate limit reached. "
                            f"Please wait ~{wait}s or create a new API key "
                            f"at aistudio.google.com/apikey."
                        )
                        answer_placeholder.warning(answer)

                except Exception as e:
                    answer = f"⚠️ Unexpected error: {str(e)}"
                    answer_placeholder.error(answer)
                    break

        # Step 4 — Show sources
        seen   = set()
        unique = []
        for s in chunks:
            key = (s["doc_name"], s["page"])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        if unique:
            with st.expander("📎 Sources"):
                for s in unique:
                    st.caption(f"**{s['doc_name']}** — page {s['page']}")
                    st.text(
                        s["text"][:300] + "…"
                        if len(s["text"]) > 300
                        else s["text"]
                    )

    # Step 5 — Save to history
    if answer:
        st.session_state.history.append({
            "user":      query,
            "answer":    answer,
            "sources":   unique,
            "assistant": answer,
        })