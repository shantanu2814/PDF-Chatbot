# 💬 Interactive PDF Chatbot (RAG)

An interactive, multi-turn conversational web application built with **Streamlit**, **LangChain**, **FAISS**, and **OpenAI**. Drag and drop single or multiple PDF documents into the interface, index them instantly using vector embeddings, and engage in context-aware Q&A with precise source citations.

---

## 📋 Table of Contents
- [Features](#-features)
- [Architecture & Workflow](#-architecture--workflow)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Features

- **📄 Multi-PDF Support:** Upload one or multiple PDF documents simultaneously through an intuitive drag-and-drop interface.
- **💬 History-Aware Conversational Memory:** Handles complex follow-up questions by recontextualizing queries against prior chat history.
- **⚡ In-Memory FAISS Vector Index:** Fast, lightweight vector similarity search powered by OpenAI `text-embedding-3-small`.
- **🔍 Granular Source Attribution:** Expandable drawer showing exact document names and page numbers used to answer every question.
- **🎨 Sleek Streamlit UI:** Production-ready web interface with real-time feedback, status spinners, and responsive layouts.

---

## 🏗️ Architecture & Workflow

```
[ User Uploads PDFs ] 
        │
        ▼
[ PyPDFLoader + Recursive Text Splitter ]
        │
        ▼
[ OpenAI Embedding Model ] ──► [ FAISS Vector Store ]
                                       │
[ User Chat Query ]                    │
        │                              │
        ▼                              │
[ History-Aware Rephraser ]            │
        │                              │
        ▼                              ▼
[ Rephrased Query ] ──────────► [ Similarity Search (Top-K) ]
                                       │
                                       ▼
                             [ Retrieved Context ]
                                       │
                                       ▼
                             [ GPT-4o Synthesis ]
                                       │
                                       ▼
                          [ Response + Source Citations ]
```

---

## 🛠️ Tech Stack

- **UI Framework:** [Streamlit](https://streamlit.io/)
- **RAG Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM:** [OpenAI GPT-4o](https://platform.openai.com/)
- **Embedding Model:** `text-embedding-3-small`
- **Vector Database:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **PDF Parser:** `pypdf` / `PyPDFLoader`
- **Language:** Python 3.10+

---

## 📁 Project Structure

```text
pdf-rag-chatbot/
├── app.py                 # Streamlit web application & RAG chain logic
├── requirements.txt       # Python dependencies
├── .env.example           # Template for environment variables
├── .gitignore             # Files ignored by git
└── README.md              # Project documentation
```

---

## 🔑 Prerequisites

- **Python 3.10** or higher
- An active **OpenAI API Key** with access to `gpt-4o` and `text-embedding-3-small`

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/pdf-rag-chatbot.git
   cd pdf-rag-chatbot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and enter your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-proj-your-actual-api-key-here
   ```

---

## 💻 Usage

1. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Open the App:**
   Your web browser will open automatically at `http://localhost:8501`.

3. **Chat with your PDFs:**
   - Drag and drop your PDF files into the sidebar upload box.
   - Click **"Process Documents"** to generate vector embeddings.
   - Type questions into the chat box at the bottom.
   - Expand the **"🔍 View Referenced Sources"** section below any response to inspect page references.

---

## 🧠 How It Works

1. **Document Loading & Chunking:**
   PDFs are parsed page-by-page using `PyPDFLoader`. Text is split into chunks of 1,000 characters with a 200-character overlap using `RecursiveCharacterTextSplitter`.

2. **Vector Indexing:**
   Chunks are embedded into 1536-dimensional vectors using OpenAI's `text-embedding-3-small` model and indexed in an in-memory `FAISS` vector store.

3. **Conversational Re-retrieval:**
   When you ask a follow-up question (e.g., *"Can you summarize its third point?"*), a history-aware retriever uses `GPT-4o` to rewrite the question into a standalone query before performing similarity search.

4. **Grounded Generation:**
   The top 4 context chunks are supplied to `GPT-4o` along with a system prompt requiring exact citations (file name and page number) and preventing out-of-context hallucinations.

---

## ❓ Troubleshooting

| Symptom | Cause | Resolution |
| :--- | :--- | :--- |
| `OpenAI API Key error` | `.env` file missing or invalid key | Ensure `.env` exists in root folder with a valid key. |
| Chat input says "Please upload and process documents first" | Vector store not created | Upload PDFs in the sidebar and click **Process Documents**. |
| PDF text not indexed | Document is scanned/image-only | Run PDFs through an OCR tool prior to upload. |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
