# Personal Assistant with RAG and LLM

## Overview

Personal Assistant with RAG and LLM is a multimodal AI system that combines Retrieval-Augmented Generation (RAG), semantic search, computer vision, and large language models to answer questions from research papers and images.

The assistant can:

* Extract and process text from PDF documents
* Generate semantic embeddings for efficient retrieval
* Build and query a FAISS vector database
* Analyze images using vision-language models
* Retrieve relevant document context
* Generate context-aware answers using a Large Language Model (LLM)

This project demonstrates practical applications of modern AI systems, including Natural Language Processing (NLP), Information Retrieval, Computer Vision, and Generative AI.

---

## Features

### Document Understanding

* PDF text extraction
* Intelligent text chunking
* Semantic embedding generation
* Vector-based retrieval using FAISS

### Retrieval-Augmented Generation (RAG)

* Context-aware question answering
* Semantic similarity search
* Efficient retrieval of relevant document sections

### Computer Vision

* Image processing pipeline
* Image caption generation
* Multimodal reasoning using text and images

### Large Language Model Integration

* Context-enriched prompting
* Research-oriented question answering
* Explainability through retrieved evidence

### Performance Enhancements

* Persistent FAISS index storage
* Cached document embeddings
* Logging and monitoring support

---

## System Architecture

```text
                    ┌─────────────────┐
                    │      PDF        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text Extraction │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Chunking      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Embeddings     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      FAISS      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Retrieval     │
                    └────────┬────────┘
                             │

 Image ─────► BLIP Caption ──┤

                             ▼
                    ┌─────────────────┐
                    │ Prompt Builder  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      Qwen       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Final Answer    │
                    └─────────────────┘
```

---

## Project Structure

```text
Personal_Assistant_with_RAG_and_LLM/

├── llm/
│   └── generator.py
│
├── pipeline/
│   ├── cache_pipeline.py
│   └── multimodel_pipeline.py
│
├── rag/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   └── index_manager.py
│
├── vision/
│   ├── image_processor.py
│   ├── image_search.py
│   ├── image_captioner.py
│   └── clip_model.py
│
├── utils/
│   └── logger.py
│
├── test.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Deep Learning

* PyTorch

### NLP

* Sentence Transformers
* Hugging Face Transformers

### Retrieval

* FAISS

### Computer Vision

* BLIP
* CLIP

### Language Models

* Qwen LLM

### Utilities

* NumPy
* Pickle
* Logging

---

## Installation

### Clone Repository

```bash
git clone https://github.com/luee07/Personal_Assistant_with_RAG_and_LLM.git

cd Personal_Assistant_with_RAG_and_LLM
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Place your PDF and image files inside the data directory.

Run:

```bash
python test.py
```

The pipeline will:

1. Load the PDF
2. Generate chunks
3. Create embeddings
4. Build or load the FAISS index
5. Retrieve relevant context
6. Generate image captions
7. Construct a multimodal prompt
8. Generate a final answer using the LLM

---

## Example Query

Question:

```text
What are the main challenges of federated learning in non-terrestrial networks?
```

Output:

```text
Direct Answer:
...

Supporting Evidence:
...

Important Observations:
...
```

---

## Future Improvements

* Conversational memory
* Multi-document retrieval
* Research paper comparison
* Agentic workflow support
* Web interface deployment
* Evaluation framework for retrieval quality
* Support for larger multimodal models

---

## Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Computer Vision
* Multimodal AI Systems
* Large Language Models
* AI System Design
* Research-Oriented AI Applications

---

## Author

Ravi Kumar

Undergraduate Researcher | AI & Deep Learning Enthusiast

GitHub: https://github.com/luee07
