# Enterprise RAG Pipeline (Local LLM) 🤖

A purely local, highly secure Retrieval-Augmented Generation (RAG) system built from scratch without relying on external cloud APIs like OpenAI. 

## 🚀 Overview

In enterprise environments, uploading private company data (like employee handbooks or proprietary code) to third-party APIs poses a severe security risk. This project solves that problem by running both the Document Retrieval system and the Large Language Model (LLM) entirely on local hardware.

## 🧠 Architecture & Tech Stack

- **Large Language Model:** Microsoft `phi-3` (Run locally via Ollama)
- **Vector Math:** NumPy (Cosine Similarity)
- **Document Processing:** `PyMuPDF` (for extracting text from PDFs)
- **Programming Language:** Python 3

## 📊 Methodology

Instead of using abstracted frameworks like LangChain, this project builds the underlying RAG mathematics entirely from scratch to demonstrate a deep understanding of GenAI mechanics.

1. **Chunking Strategy:** Implemented overlapping text chunking to ensure semantic context is not lost at the boundaries of paragraphs.
2. **Mathematical Retrieval:** Instead of relying on vector databases, this project calculates **Cosine Similarity** directly using NumPy arrays to identify the Top-K most relevant document chunks based on the user's query.
3. **Prompt Engineering:** Engineered strict system prompts to force the LLM to answer *only* using the provided context, eliminating hallucinations.

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install ollama pymupdf numpy
   ```
2. **Install Local LLM Server:**
   Download and install [Ollama](https://ollama.com/), then pull the model:
   ```bash
   ollama pull phi3
   ```
3. **Run the RAG System:**
   Place your private PDF in the root directory and execute:
   ```bash
   python local_rag.py
   ```

---
*Developed to demonstrate advanced Prompt Engineering, applied linear algebra, and strict data privacy compliance in GenAI.*
