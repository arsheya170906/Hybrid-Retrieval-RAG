# Hybrid Retrieval RAG Pipeline

A robust, modern RAG (Retrieval-Augmented Generation) pipeline for querying PDF documents. Drop in a PDF, run the ingest step once, and ask questions in plain language. 

The pipeline uses a **Hybrid Retrieval** approach (Dense Embeddings via ChromaDB + Sparse Keyword matching via BM25), re-ranked with a Cross-Encoder, and generates answers using Mistral AI via **LangChain**.

## Architecture

![Hybrid RAG Architecture](https://i.postimg.cc/3wYktT2x/Hybrid-RAG.jpg)

## How It Works

- **Ingestion (`ingest.py`)**: Reads the PDF, chunks the text, and stores the data in two formats:
  - `chroma_db/`: A local vector store containing dense embeddings (using `all-MiniLM-L6-v2`) for semantic search.
  - `bm25_corpus.pkl`: A raw text list for sparse, exact-keyword matching.
  
- **Querying (`query.py`)**: Runs both dense and sparse retrieval simultaneously. It merges the results, scores them for exact relevance using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), and passes the best context to the Mistral LLM to generate your answer.

## Tech Stack

| Component | Library |
|---|---|
| PDF parsing | `pypdf` |
| Dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse retrieval | `rank_bm25` via LangChain |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | `langchain_chroma` (ChromaDB) |
| LLM Framework | `langchain` |
| LLM | `mistral-large-latest` via Mistral AI API |

## Project Structure

```
Hybrid-Retrieval-RAG/
├── ingest.py           # PDF parsing, chunking, embedding, storage
├── query.py            # Hybrid retrieval, reranking, answer generation
├── AWS.pdf             # Source document
├── chroma_db/          # Persisted vector store (auto-created on first run)
├── bm25_corpus.pkl     # Persisted sparse corpus (auto-created on first run)
└── .env                # API keys
```

## How To Run It

### 1. Setup & Install Dependencies
Make sure your terminal is using your target Python environment, then install the required packages:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-chroma langchain-huggingface langchain-mistralai pypdf sentence-transformers rank-bm25 python-dotenv
```

### 2. Configure API Key
Add your Mistral API key to the `.env` file in the project folder:
```
MISTRAL_API_KEY=your_key_here
```

### 3. Build the Database (Run Once)
Run the ingestion script to process the PDF and build the `chroma_db` and `bm25_corpus.pkl` files:
```bash
python ingest.py
```

### 4. Ask Questions
Start the query script to enter the interactive chat prompt:
```bash
python query.py
```
Type your questions and press Enter. Type `quit` to exit the loop.

To re-index a new PDF, just delete the `chroma_db/` folder and `bm25_corpus.pkl`, drop in your new PDF (updating the filename in `ingest.py`), and run `ingest.py` again!
# Hybrid-Retrieval-RAG
