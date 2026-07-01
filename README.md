# 🤖 Chat with Website — RAG-based Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you chat with the content of any website. Paste a URL, and the app scrapes the page, indexes it into a vector database, and answers your questions using only that website's content — powered by open-source models via Hugging Face.

---

## 📖 Overview

This app combines three pieces into a working RAG pipeline:

1. **Web scraping** — pulls text content from a given URL
2. **Vector search (retrieval)** — finds the most relevant chunks of that content for a user's question
3. **LLM generation** — an open-source LLM answers the question using only the retrieved chunks as context

This means answers come from the actual website content, not just the model's general training data.

---

## ✨ Features

- 🔗 Chat with any website by pasting its URL
- 🧠 History-aware retrieval — understands follow-up questions like *"tell me more about that"*
- 📚 Answers grounded in the website's actual content (RAG)
- 🗂️ Automatically re-indexes when you switch to a new URL
- 💬 Persistent chat interface built with Streamlit

---

## 🏗️ Architecture

```
User enters URL
      ↓
WebBaseLoader scrapes the page
      ↓
RecursiveCharacterTextSplitter splits it into chunks
      ↓
HuggingFace Embeddings convert chunks → vectors
      ↓
Chroma vector store indexes the vectors
      ↓
User asks a question
      ↓
History-aware retriever rewrites the question using chat history
      ↓
Similarity search retrieves the most relevant chunks
      ↓
LLM (Qwen3-4B-Instruct via HuggingFace) generates an answer using those chunks
      ↓
Answer displayed in the chat UI
```

---

## 🧰 Tech Stack

| Component        | Tool/Library                              |
|-------------------|--------------------------------------------|
| UI                | [Streamlit](https://streamlit.io)          |
| Orchestration     | [LangChain](https://www.langchain.com)     |
| Vector Store      | [ChromaDB](https://www.trychroma.com)      |
| Embeddings        | `BAAI/bge-small-en-v1.5` (via HuggingFace) |
| LLM               | `Qwen/Qwen3-4B-Instruct-2507` (via HuggingFace) |
| Web Scraping      | LangChain `WebBaseLoader`                  |

---

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # Local environment variables (NOT committed)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup (Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_api_token_here
```

Get a free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🚀 Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (make sure `.env` and `chroma_db/` are **not** committed — see `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"Create app"** → select your repository, branch, and `app.py` as the entry point.
4. In **Advanced settings → Secrets**, add:
   ```toml
   HF_TOKEN = "your_huggingface_api_token_here"
   ```
5. Click **Deploy**. Your app will be live at a `*.streamlit.app` URL within a few minutes.

> ⚠️ **Note:** Streamlit Community Cloud has an ephemeral filesystem — the local Chroma vector store resets on app restarts. This is expected behavior; the app re-indexes automatically whenever a URL is entered.

---

## 🔑 Environment Variables

| Variable   | Description                                      |
|------------|---------------------------------------------------|
| `HF_TOKEN` | Your Hugging Face API access token (required)     |

---

## 🛣️ Roadmap / Possible Improvements

- [ ] Clean scraped content to remove nav bars/footers/cookie banners
- [ ] Add source citations to answers
- [ ] Support multiple URLs per session
- [ ] Swap in a reranker for improved retrieval accuracy
- [ ] Add hard word-limit enforcement on generated answers

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).