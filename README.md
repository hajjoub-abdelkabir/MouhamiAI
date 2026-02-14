# ⚖️ MouhamiAI: Moroccan Family Code Legal Assistant (RAG System)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-⚡-green)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Groq](https://img.shields.io/badge/LLM-Groq_(Llama_3)-black)

**MouhamiAI** is an advanced Retrieval-Augmented Generation (RAG) system specifically designed to act as a legal assistant for the Moroccan Family Code (Moudawana). It leverages state-of-the-art NLP models and a robust data pipeline to provide accurate, context-aware, and strictly legal responses in Arabic.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Tech Stack](#-tech-stack)
4. [Directory Structure](#-directory-structure)
5. [Installation & Setup](#-installation--setup)
6. [Usage](#-usage)
7. [Security & Guardrails](#-security--guardrails)

---

## 🚀 Project Overview
Navigating legal texts can be complex. MouhamiAI simplifies this by allowing users to ask natural language questions (in standard Arabic or Moroccan Darija) about family law (marriage, divorce, custody, inheritance). 

The system ensures high accuracy by strictly retrieving information from the official digitized PDF of the Moudawana and explicitly citing the article numbers in its responses. It includes a semantic guardrail to prevent the AI from answering non-legal queries.

---

## 🧠 System Architecture

The project is divided into two main pipelines:

### 1. Offline Data Ingestion & Indexing Pipeline (Preparation Phase)
This phase transforms raw, unstructured PDF data into a highly structured, searchable vector database.
* **Text Extraction (OCR):** Utilizes `pdf2image` and `pytesseract` to extract Arabic text from scanned PDF pages.
* **Semantic Chunking & Hierarchical Parsing (`ingest.py`):** Instead of naive character-count splitting, the system uses custom Regular Expressions (`re`) to split the text precisely by legal articles (e.g., "المادة 1"). It also intelligently extracts the hierarchical context (Book -> Section -> Chapter) and attaches it as metadata to each article, saving the output as a structured `JSON` file.
* **Vectorization (`vectorize.py`):** Reads the JSON file, combines the hierarchical context with the article text, and embeds it using the `intfloat/multilingual-e5-large` model. The resulting dense vectors are persistently stored in a local **ChromaDB**.

### 2. Online RAG Inference Pipeline (Runtime Phase)
This is the interactive phase powered by `app.py`.
* **Semantic Guardrail (Intent Router):** Before any retrieval happens, the user's query is analyzed. If the query is outside the legal domain (e.g., sports, general chat), the system instantly rejects it, saving compute resources and ensuring professional behavior.
* **Query Reformulation:** If part of an ongoing conversation, the system contextualizes the user's question based on chat history to formulate a standalone search query.
* **Vector Retrieval:** Queries ChromaDB for the top 7 most semantically similar chunks.
* **Cross-Encoder Reranking:** Passes the initial results through `Omartificial-Intelligence-Space/ARA-Reranker-V1` to re-score and select the top 4 most highly relevant articles, dramatically reducing hallucinations.
* **Response Generation:** Injects the reranked context into a strict Prompt Template and generates the final Arabic response using the **Llama-3.3-70b-versatile** model via the **Groq API** for ultra-fast inference.

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Customized with RTL CSS and Arabic typography)
* **Orchestration:** LangChain
* **Embeddings:** HuggingFace (`multilingual-e5-large`)
* **Reranker:** Sentence-Transformers (`ARA-Reranker-V1`)
* **Vector Database:** ChromaDB (Local SQLite)
* **LLM:** Groq API (Llama 3.3)
* **Containerization:** Docker

---

## 📁 Directory Structure

```text
MOUHAMI_AI/
├── app.py                      # Main Streamlit application (Online RAG pipeline)
├── ingest.py                   # PDF text extraction and hierarchical JSON structuring
├── vectorize.py                # Embedding generation and ChromaDB population
├── cli_app.py                  # CLI version of the app for fast terminal testing
├── debug.py                    # PyMuPDF debugging script for raw text inspection
├── moudawana_articles.json     # Cleaned, structured data output from ingest.py
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration for isolated deployment
└── .gitignore                  # Git ignore rules (protects .env and DB)