
# Hospital Protocol RAG Conversational QA System

This repository contains an end-to-end Retrieval-Augmented Generation (RAG) system designed for conversational question answering over hospital treatment protocols and related healthcare documents. Leveraging LangChain, FAISS, Google Gemini, and HuggingFace models, it supports multi-turn interactive QA with contextual memory and semantic retrieval.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed File Descriptions](#detailed-file-descriptions)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Healthcare professionals need fast, accurate access to complex treatment protocols, drug information leaflets, and insurance policies. This project implements a conversational QA system that enables users to ask multiple interactive questions referencing prior answers, with the system retrieving relevant document chunks and generating concise, context-aware responses.

Built with modern NLP tools and APIs, it seamlessly switches between Google Gemini generative models, local HuggingFace transformers, and dummy fallback models to ensure continuous availability.

---

## Features

- **Text Extraction and Chunking**: Loads and parses large PDF documents, splitting them into overlapping text chunks for fine-grained retrieval.
- **Hybrid Embedding Options**: Supports Google Generative Language API embeddings and local HuggingFace embeddings with automated fallback.
- **FAISS Vector Store**: Builds and caches a fast semantic search index to efficiently retrieve relevant document chunks.
- **Conversational Memory**: Tracks multi-turn conversation history for interactive, context-rich question answering.
- **MultiQueryRetriever**: Enhances retrieval by reformulating follow-up queries considering dialog context.
- **Multi-Model Answering**: Uses Google Gemini LLMs when available, or falls back on local transformer models or simple dummy responses.
- **Source Attribution**: Returns the sources of answers so users can verify information in original documents.
- **Command-Line and Interactive Modes**: Supports scripted and interactive question-answer sessions.

---

## File Structure

├── .env # Environment variables (API keys, flags)
├── loader.py # Loads and parses PDFs into text chunks
├── list_model.py # Utility to list available Google generative AI models
├── Vector_store.py # Creates vector embeddings and builds FAISS index
├── qa_system.py # Single-turn QA pipeline integrating retriever and LLMs
├── conversational_qa.py # Multi-turn conversational QA with memory and query reformulation
├── standard-treatment-guidelines.pdf # Sample hospital protocol PDF document
├── README.md # This file

text

---

## Installation

1. Clone this repository:

git clone <repository-url>
cd <repository-directory>

text

2. Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate # Unix/macOS
.\venv\Scripts\activate # Windows

text

3. Install required Python packages:

pip install -r requirements.txt

text

4. Create a `.env` file in the root directory with the following (replace with your API key):

GOOGLE_API_KEY=your_google_api_key_here
USE_GOOGLE=1
GOOGLE_MODEL=models/gemini-1.5-pro

text

5. Place your PDF file(s) (e.g., `standard-treatment-guidelines.pdf`) in the project directory or update the PDF path in `loader.py`.

---

## Usage

### Step 1: Load and Parse PDF

python loader.py

text

- Confirms PDF loading and parsing by printing sample content.

### Step 2: Generate Vector Embeddings & Build Index

python Vector_store.py

text

- Splits documents, generates embeddings (Google or local), and saves a FAISS index.

### Step 3: Single-turn QA

python qa_system.py

text

- Runs an example query with QA over the loaded index, using best available LLM.

### Step 4: Multi-turn Conversational QA

python conversational_qa.py

text

- Runs an interactive or scripted two-turn conversational demo with query reformulation and memory.

Example scripted run:

python conversational_qa.py --q1 "What is the recommended hypertension treatment for patients over 60?" --q2 "Compare it with treatment for patients under 40."

text

---

## Detailed File Descriptions

### `loader.py`

- Loads hospital protocols PDF(s) via `PyPDFLoader`.
- Suppresses verbose PDF parsing logs for clarity.
- Returns parsed document texts for chunking.

### `list_model.py`

- Queries Google Generative Language API to list accessible models for your API key.
- Useful for verifying API/configuration.

### `Vector_store.py`

- Uses `RecursiveCharacterTextSplitter` to split text into overlapping chunks.
- Attempts embedding with Google Generative API embeddings; falls back gracefully to local HuggingFace MiniLM embeddings if needed.
- Indexes embedded chunks with FAISS for fast similarity search.
- Saves index locally for persistence.

### `qa_system.py`

- Loads FAISS index and uses HuggingFace embeddings for retrieval.
- Attempts to use Google Gemini generative LLM; falls back gracefully to local HuggingFace generation or dummy responses.
- Performs single-turn question answering and prints results with source metadata.
- Implements robust API error handling and quota checks.

### `conversational_qa.py`

- Implements multi-turn conversational QA with context memory using LangChain’s `ConversationBufferMemory`.
- Wraps retrieval with `MultiQueryRetriever` for query reformulation based on conversation history.
- Supports hierarchical retrieval via optional `ParentDocumentRetriever`.
- Uses Google Gemini or local HuggingFace LLMs with fallback to dummy responses if needed.
- Interactive and scripted 2-turn demo for demonstration and testing of advanced conversational features.

---

## Troubleshooting

- **API key missing or invalid**: Ensure `.env` contains a valid Google API key. If quota is exceeded or key invalid, the system automatically falls back to local or dummy models.
- **Out of disk space on Windows**: HF model cache is redirected to `D:/hf_cache` if available to prevent C: drive overload.
- **PDF parsing issues**: Only text-based PDFs are supported. For scanned PDFs, use OCR preprocessing.
- **Dependency issues**: Verify all requirements installed at matching versions (`requirements.txt`). HuggingFace and PyTorch are required for local generation.

---

## Contributing

Feel free to submit issues, fork the repository, and create pull requests. Your feedback and contributions are welcome to improve functionality and documentation.

---

## License

No Lisence

---

## Contact

Jatin Kanyan
Jatinkanyan11@gmail.com
9306980229

---

_Thank you for exploring the Hospital Protocol Conversational QA System!_
