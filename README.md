# 📄 Ask My PDF Bot

An AI-powered document assistant that lets you have natural conversations with your PDF and Word documents. Built with a production-grade RAG (Retrieval-Augmented Generation) pipeline featuring hybrid search, source citations, and multi-LLM support.

---

## 🚀 Live Demo

🔗 [Ask My PDF Bot on Streamlit Cloud](https://your-app-link.streamlit.app)

---

## 📌 Features

- ✅ **No hallucination** — answers strictly from uploaded documents only
- ✅ **Source citations** — every answer shows document name and page number
- ✅ **Multi-document support** — upload and query across multiple PDFs simultaneously
- ✅ **Conversational memory** — maintains context across multiple turns
- ✅ **Hybrid retrieval** — combines FAISS semantic search + BM25 keyword search via RRF
- ✅ **Multi-LLM support** — switch between Groq, Gemini, or local Ollama
- ✅ **Offline mode** — fully local with Ollama + Llama 3.2 (no API key needed)
- ✅ **Large PDF support** — handles 100+ page documents
- ✅ **System monitoring** — live CPU and RAM usage in sidebar

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| PDF Extraction | PyMuPDF (fitz) |
| Text Chunking | NLTK semantic chunking |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS (local) |
| Keyword Search | BM25 (rank-bm25) |
| Retrieval | Hybrid RRF (FAISS + BM25) |
| LLM (fast) | Groq — Llama 3.3 70B |
| LLM (cloud) | Google Gemini 2.0 Flash |
| LLM (offline) | Ollama — Llama 3.2 |
| Language | Python 3.11 |

---

## 🏗️ Architecture
PDF/Word Upload
│
▼
PyMuPDF Extraction (page-level)
│
▼
NLTK Semantic Chunking (sentence boundaries)
│
▼
BAAI/bge-small-en-v1.5 Embeddings
│
├──────────────────┐
▼                  ▼
FAISS Vector Index    BM25 Keyword Index
│                  │
└────────┬─────────┘
▼
Hybrid RRF Retrieval
│
▼
Top-K Chunks + Metadata
(doc name + page number)
│
▼
Prompt Builder + Chat History
│
▼
LLM (Groq / Gemini / Ollama)
│
▼
Answer + Source Citations

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/OP2004/ask-my-pdf-bot.git
cd ask-my-pdf-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

### 3. Set up API keys
Create a `.env` file in the root folder:
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

Get your free API keys here:
- Groq (fastest): [console.groq.com](https://console.groq.com)
- Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔌 Offline Mode (No API Key Required)

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the model:
```bash
ollama pull llama3.2
```
3. In the app sidebar, select **Ollama (offline)**

---

## 📁 Project Structure
ask-my-pdf-bot/
├── app.py                  # Streamlit UI
├── rag/
│   ├── init.py
│   ├── ingestor.py         # PDF loading, chunking, embedding
│   ├── retriever.py        # Hybrid FAISS + BM25 retrieval
│   └── generator.py        # LLM generation + prompt builder
├── requirements.txt
└── README.md

---

## 💡 How It Works

1. **Upload** your PDF or Word documents
2. **Indexing** — documents are extracted, chunked by sentence boundaries, embedded with BGE, and stored in FAISS + BM25
3. **Ask** a question in natural language
4. **Retrieval** — hybrid search finds the most relevant chunks using both semantic similarity and keyword matching
5. **Generation** — LLM answers strictly from retrieved context with inline citations
6. **Citations** — every answer shows the source document and page number

---

## 🏭 Industry Use Cases

- ⚖️ **Legal** — contract analysis, case law research, policy Q&A
- 🏥 **Healthcare** — clinical guidelines, patient record querying
- 🎓 **Education** — research paper Q&A, textbook assistant
- 🏢 **Enterprise** — HR policy assistant, IT knowledge base
- 🏛️ **Government** — public records querying, regulatory documents

---

## 🔮 Future Improvements

- [ ] Agentic RAG with multi-agent workflows
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] ReRanking with cross-encoders
- [ ] Voice interface (speech-to-text + text-to-speech)
- [ ] Async processing with Celery + Redis
- [ ] AWS S3 document storage

---

## 👨‍💻 Author

**OP2004**  
[GitHub](https://github.com/OP2004)

---

## 📄 License

MIT License — free to use and modify.