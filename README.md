# FAQ Chatbot

A **multi-turn RAG chatbot** that answers students' questions about the DataTalksClub courses (LLM Zoomcamp, ML Zoomcamp, Data Engineering Zoomcamp and more). Its answers come from the official course FAQ.

Ask *"Can I still join the course?"* and then a follow-up such as *"and will I get a certificate that way?"*. The bot understands what "that way" refers to, finds the right FAQ entries with **hybrid search** (keyword + semantic vectors), answers from them, and shows its sources.

Built during **LLM Zoomcamp, Modules 1–2: RAG and Vector Search**.

---

## Features

- **Answers grounded in the FAQ.** The bot answers only from retrieved FAQ entries and says *"I don't know"* when the FAQ doesn't cover the question.
- **Hybrid search.** BM25 keyword search finds exact terms such as error messages and tool names. Vector search with `all-MiniLM-L6-v2` finds answers worded differently from the question: *"when is it due?"* → *"capstone project deadline"*.
- **Multi-turn conversations.** The LLM sees the recent conversation, so follow-ups like *"and is it free?"* are understood.
- **Fast.** Follow-up answers take about 0.6–0.8 s, and a cold start takes about 0.5 s (see [Performance](#performance)).
- **6 courses, 1,427 FAQ entries.** A course switcher in the UI limits search to the selected course.
- **Transparent sources.** Every answer lists the FAQ entries it used, the exact query that was searched and the search mode.
- **Guardrails.** It refuses prompt-injection attempts and off-topic requests.
- **Evaluation suite.** It measures retrieval and answer quality across conversation turns and compares keyword, vector and hybrid search.
- **Serverless deployment** on Vercel.

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| **Architecture** | RAG (Retrieval-Augmented Generation) | retrieve → augment prompt → generate |
| **LLM provider** | [Groq](https://groq.com) | Fast inference through an OpenAI-compatible API |
| **LLMs** | `openai/gpt-oss-20b` | Writes the answers (and optionally rewrites follow-up questions) |
| | `openai/gpt-oss-120b` | LLM-as-a-judge for evaluation |
| **LLM SDK** | `openai` Python SDK | Pointed at Groq's `base_url` |
| **Embedding model** | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384-dim sentence embeddings, pinned to a fixed revision |
| **Embedding library** | [sentence-transformers](https://huggingface.co/sentence-transformers) (ONNX backend) | Turns all 1,427 FAQ entries into vectors (`data/embeddings.npy`) |
| **Model file** | MiniLM's official 8-bit ONNX export (`onnx/model_quint8_avx2.onnx`, 23 MB) | Bundled with the app: no download at startup. Same retrieval quality as the 90 MB full-precision model in our evaluation |
| **Query encoder (in the app)** | The same ONNX file, run with `onnxruntime` + `tokenizers` | Produces the same vectors as sentence-transformers (cosine 1.00000), with no PyTorch, so it fits on Vercel |
| **Vector search** | NumPy exact cosine similarity | 1,427 × 384 matrix (2 MB); brute force takes under 1 ms, so no vector DB is needed |
| **Keyword search** | **BM25**, written in pure Python | Field boosts: question ×2, section ×0.5 |
| **Hybrid fusion** | Reciprocal Rank Fusion (RRF) | Merges the keyword and vector rankings |
| **Retrieval (notebooks)** | `minsearch` (TF-IDF and `VectorSearch`) | In-memory indexes for learning |
| **Database** | **SQLite** via `sqlitesearch` | Persistent full-text index in `database/faq.db` |
| **Backend** | FastAPI + Pydantic, served by Uvicorn | Stateless REST API |
| **Frontend** | HTML, CSS and vanilla JavaScript | Single page, light and dark mode, mobile-friendly |
| **Deployment** | Vercel (Python serverless functions) | |
| **Evaluation** | Synthetic test conversations, Hit Rate@5, MRR, LLM-as-a-judge | |
| **Tooling** | `uv` (environment), `pytest` (tests), `ruff` (lint), Jupyter | |

---

## How it works

```
                    ┌───────────────────────────── one chat turn ─────────────────────────────┐
user message  ───►  │ 1. RETRIEVE  • BM25 keyword search                          (~1 ms)     │
+ chat history      │              • vector search: embed the question with MiniLM,           │
                    │                cosine similarity over 1,427 FAQ vectors     (~5 ms)     │
                    │              → filtered by course, merged with RRF → top 5              │
                    │ 2. AUGMENT   prompt = rules + last 6 messages + question + top-5 FAQs   │
                    │ 3. GENERATE  LLM writes the answer (Groq)                 (~0.6 s)      │
                    └─────────────────────────────────────────────────────────────────────────┘
                     optional (REWRITE_QUERIES=true): before step 1, an extra LLM call
                     rewrites a follow-up into a standalone query (+~1 s)
                                                   │
                                                   ▼
                              answer + sources + searched query + search mode
```

**Two stages for embeddings**

| | Offline (your machine) | Online (the app) |
|---|---|---|
| What | Embed all FAQ entries | Embed the user's question |
| Tool | sentence-transformers, ONNX backend | ONNX Runtime + tokenizers (no PyTorch) |
| Model file | `models/all-MiniLM-L6-v2/model_quint8_avx2.onnx` | the same file |
| Output | `chatbot/data/embeddings.npy` | One 384-dim vector per query |

Both stages use the same model file, and `build_embeddings.py` checks that they produce identical vectors. Texts are embedded **one at a time**. With an 8-bit model, a text's vector would otherwise shift slightly depending on which other texts share its batch.

The server is **stateless**. The browser keeps the conversation and sends it with each message, which is how serverless functions need to work.

---

## Project structure

```
.
├── chatbot/                     THE APP
│   ├── api/index.py             REST API (FastAPI): /api/chat, /api/courses, /api/health
│   ├── faqbot/                  chatbot logic
│   │   ├── config.py              settings: models, paths, search mode, limits
│   │   ├── llm.py                 Groq client
│   │   ├── prompts.py             all prompts in one place
│   │   ├── data.py                loads FAQ data + embeddings, builds the indexes
│   │   ├── search.py              BM25 keyword search
│   │   ├── vector_search.py       vector (cosine similarity) search
│   │   ├── embeddings.py          MiniLM query encoder (ONNX Runtime)
│   │   ├── model_files.py         re-downloads the model files if they're missing
│   │   ├── rag.py                 ConversationalRAG: hybrid retrieve → answer
│   │   └── bot.py                 create_bot(): assembles the bot, falls back to BM25 if needed
│   ├── public/                  served by the CDN: index.html (chat UI) + courses.json
│   ├── data/                    what the app searches
│   │   ├── documents.json         1,427 FAQ entries
│   │   ├── courses.json           course list
│   │   ├── embeddings.npy         1,427 × 384 MiniLM vectors (built by sentence-transformers)
│   │   └── embeddings.json        model, revision and row order of the vectors
│   ├── models/                  MiniLM ONNX model (23 MB) + tokenizer, bundled with the app
│   ├── evaluation/              multi-turn evaluation (test datasets, scripts, results)
│   ├── scripts/
│   │   ├── build_data.py          refreshes data/documents.json from the FAQ website
│   │   └── build_embeddings.py    embeds the FAQ with sentence-transformers
│   ├── tests/                   automated tests (offline, no API calls)
│   ├── requirements.txt         production dependencies (installed by Vercel)
│   └── vercel.json              deployment config
│
├── notebooks/                   LEARNING PATH: how the RAG was built, step by step
│   ├── 01_rag_from_scratch.ipynb
│   ├── 02_ingest_in_memory.ipynb
│   ├── 03_rag_class.ipynb
│   ├── 04_persistent_sqlite_index.ipynb
│   └── 05_vector_search.ipynb   sentence-transformers + MiniLM, keyword vs vector
│
├── src/llm_zoomcamp_dtc/        reusable code the notebooks import (ingest, RAG class, SQLite index)
├── database/                    SQLite search index faq.db (generated, not committed)
│
├── .env.example                 template for your API key → copy to .env
├── pyproject.toml               dependencies + tool settings (one environment for everything)
├── uv.lock                      exact dependency versions
└── .gitignore, .gitattributes, .editorconfig, .python-version
```

---

## Getting started

### 1. Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python package manager. It installs the right Python version for you.
- A free Groq API key from https://console.groq.com/keys
- Node.js, only if you want to deploy to Vercel.

### 2. Install

```bash
git clone <this-repo>
cd "01. Build Your First RAG Application with LLMs"
uv sync
```

`uv sync` creates **one** virtual environment (`.venv/`) with everything: the chatbot, sentence-transformers (CPU-only PyTorch), the notebooks, tests and lint.

### 3. Add your API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Run the chatbot

```bash
uv run uvicorn api.index:app --app-dir chatbot --reload
```

Open **http://localhost:8000** and start asking questions. Press `Ctrl+C` to stop the server.


---

## Everyday commands

Run all commands from the project root.

| What | Command |
|---|---|
| Run the chatbot | `uv run uvicorn api.index:app --app-dir chatbot --reload` |
| Interactive API docs | Start the chatbot, then open http://localhost:8000/docs |
| Run the tests | `uv run pytest` |
| Run one test file | `uv run pytest chatbot/tests/test_vector_search.py` |
| Lint the code | `uv run ruff check .` |
| Open the notebooks | `uv run python -m jupyter lab` (or open them in VS Code and pick the `.venv` kernel) |
| Build the SQLite index | `uv run build-faq-db` (add `--course all` for every course) |
| Refresh the FAQ data | `uv run python chatbot/scripts/build_data.py` |
| Re-embed the FAQ | `uv run python chatbot/scripts/build_embeddings.py` (**required after refreshing the data**) |

### Use the API directly

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "and is it free?",
    "history": [
      {"role": "user", "content": "Do I have to use OpenAI for the homework?"},
      {"role": "assistant", "content": "No, you can use other providers like Groq."}
    ],
    "course": "llm-zoomcamp"
  }'
```

The response contains `answer`, `standalone_question` (what was actually searched), `search_mode` and `sources`.

---

## Configuration

All settings are read from `.env`. Only the API key is required.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — (required) | Groq API key |
| `SEARCH_MODE` | `hybrid` | `hybrid` (keyword + vector), `bm25` (keyword only) or `vector` (semantic only) |
| `CHAT_MODEL` | `openai/gpt-oss-20b` | Model that writes the answers |
| `REWRITE_QUERIES` | `false` | `true` adds an LLM call that rewrites follow-ups into standalone search queries (+~1 s per answer, no retrieval gain in our evaluation) |
| `REWRITE_MODEL` | same as `CHAT_MODEL` | Model used when `REWRITE_QUERIES=true` |
| `JUDGE_MODEL` | `openai/gpt-oss-120b` | Model that grades answers during evaluation |

The embedding model (`all-MiniLM-L6-v2`) and its pinned revision are set in `chatbot/faqbot/config.py`. If you change them, re-run `build_embeddings.py`.

---

## Deploy to Vercel

```bash
cd chatbot
npx vercel          # first time: log in and link the project
npx vercel --prod   # deploy to production
```

Then in the Vercel dashboard, open **Settings → Environment Variables**, add `GROQ_API_KEY`, and redeploy.

If you deploy from GitHub instead, import the repo in Vercel and set **Root Directory** to the `chatbot` folder.

**How vector search fits on Vercel**
- Vercel installs only `chatbot/requirements.txt`: FastAPI, OpenAI, NumPy, ONNX Runtime and tokenizers, with **no PyTorch**.
- The function bundle is about **204 MB**: 178 MB of Linux packages plus 26 MB of code, data, embeddings and the 23 MB model. That's under Vercel's ~250 MB limit.
- If vector search ever fails to load, the bot logs a warning and keeps answering with keyword search only.

Tests, evaluation and scripts are excluded from the deployment by `chatbot/.vercelignore`. Vercel runs Python 3.12 (`chatbot/.python-version`), and the app code is compatible with it.

---

## Performance

Measured locally against the real Groq API:

| Step | Time |
|---|---|
| Open the page (static HTML + `courses.json` from the CDN) | no Python involved |
| Cold start: load model + build BM25 and vector indexes | **~0.5 s** |
| Embed a question (MiniLM, ONNX) | ~5 ms |
| Hybrid search (BM25 + vector + fusion) | ~10 ms |
| Follow-up answer, end to end | **~0.6–0.8 s** |
| First question (includes opening the Groq connection) | ~1.5 s |

What makes it fast:

1. **The page never waits for Python.** `index.html` and `courses.json` are static files on Vercel's CDN.
2. **Warm-up on page open.** The page calls `/api/warmup` as soon as it loads, so a cold serverless instance starts while the user is still typing.
3. **No model download.** The 23 MB 8-bit model ships inside the function. The 90 MB full-precision model would have needed a ~20 s download on every cold start.
4. **One LLM call per answer.** Query rewriting is off by default; it doubled response time without improving retrieval.
5. **No vector database round-trip.** Vectors live in memory; exact search over 1,427 vectors takes under 1 ms.

**Optional: keep it warm.** Vercel reuses warm instances between requests. If the site gets little traffic, a free uptime monitor (e.g. UptimeRobot) can call `https://<your-app>.vercel.app/api/health` every 5 minutes, so even the first visitor gets a warm instance.

---

## Evaluation

Ordinary RAG evaluations test one question at a time. Chatbots usually fail later in a conversation: on a follow-up like *"and when is that due?"*, a topic switch, or when an early mistake carries through. The evaluation suite tests these cases.

Run the scripts from inside the `chatbot` folder:

```bash
cd chatbot
uv run python -m evaluation.generate_conversations --n 30   # 1. build test conversations
uv run python -m evaluation.evaluate_retrieval              # 2. is the right FAQ entry found?
uv run python -m evaluation.evaluate_answers --limit 10     # 3. are the answers good?
```

| Step | What it measures |
|---|---|
| **1. Test data** | An LLM writes 2–3-turn conversations from real FAQ entries. Each follow-up depends on earlier turns, and each turn is labelled with the FAQ entry that answers it. Four hand-written edge cases are added: a topic switch, a chain of "it/that" follow-ups, a prompt-injection attempt and an off-topic question. |
| **2. Retrieval** | **Hit Rate@5** (is the right entry in the top 5?) and **MRR** (how high is it ranked?), per turn. Compares **BM25, vector and hybrid** search, each with the raw message alone and with raw + rewritten queries. |
| **3. Answers** | The bot holds full conversations using its own previous answers. A stronger LLM judge then scores each turn from 1 to 5 on **relevance, groundedness, context use, correctness and safety**, and each conversation on **coherence** and **task completion**. Use `--search-mode bm25` to compare against keyword-only search. |

Results are saved to `chatbot/evaluation/results/`.

**Latest retrieval results** (24 conversations, 55 turns; hybrid search on the raw message is what the chatbot uses):

| Search | Hit Rate@5 (all turns) | Hit Rate@5 (follow-ups) | MRR (all turns) |
|---|---|---|---|
| BM25 keyword | 0.909 | 0.871 | 0.879 |
| Vector (MiniLM, 8-bit) | 0.927 | 0.871 | 0.845 |
| **Hybrid (BM25 + vector)** | **0.945** | **0.903** | **0.886** |

The 90 MB full-precision model gave the same hit rates (hybrid MRR 0.880), so the 23 MB model costs nothing here.

**What the evaluation found**

1. **Hybrid search finds the right entry most often.** Keyword and vector search fail on *different* questions, so combining them beats either alone.
2. **Vectors fix vocabulary gaps.** *"and for the capstone, when is that due?"* scored 0/5 with keyword search, because the FAQ says "deadline", not "due". With hybrid search the answer is found and answered correctly.
3. **Vectors alone rank slightly worse** (lower MRR). MiniLM still favours shared words: *"homework due"* sits closer to *"homework party"* (0.74) than to *"submission deadline"* (0.56). That's why vectors are fused with BM25, not used alone.
4. **Query rewriting didn't change retrieval** on this dataset, because the right entry was usually already ranked #1 from the raw message. It also cost about 1 s per answer, so it's **off by default**. The answer LLM still sees the conversation, so follow-ups are understood.
5. **Use retrieval metrics alongside the LLM judge.** The judge once gave 5/5 to an *"I don't know"* answer for a question the FAQ did cover, because retrieval had missed the entry.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Create `.env` from `.env.example` and add your key |
| `Address already in use` | Another program is using port 8000. Add `--port 8001` to the command |
| `uv: command not found` | Install uv: https://docs.astral.sh/uv/getting-started/installation/ |
| `An Application Control policy has blocked this file … torch_python.dll` | Windows Smart App Control blocked a PyTorch DLL. This project installs CPU-only PyTorch from PyTorch's own index on Python 3.14, which loads on this machine. Keep `.python-version` at `3.14` and run `uv sync` |
| `embeddings.npy is out of date` | The FAQ data changed. Run `uv run python chatbot/scripts/build_embeddings.py` |
| Log shows `Vector search unavailable; falling back to BM25` | The model files in `chatbot/models/` are missing and couldn't be downloaded. The bot still works with keyword search. Run `uv run python chatbot/scripts/build_embeddings.py` to restore them |
| Chat says *"The assistant is busy right now"* | Groq's free tier allows **200,000 tokens per day** for `gpt-oss-20b`; each answer uses roughly 1–2k, so very roughly 100–200 answers a day. Wait for the limit to reset, or upgrade to Groq's Dev tier for a public site |
| Evaluation is slow or hits `429` errors | Groq's free tier is rate-limited. The client retries automatically; use `--limit` for quick runs |
| Notebook can't import `llm_zoomcamp_dtc` | Select the project's `.venv` as the notebook kernel |
