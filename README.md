# fast-doc-rag

A lightweight and fast Flask API that supports multiple document formats (PDF, text, etc.) to compute embeddings and perform retrieval augmented generation (RAG) using OpenAI’s API. The computed embeddings are stored in a PostgreSQL database for fast lookup.

## Features

- **Document Parsing:** Supports extracting text from PDFs (and can be extended to other formats).
- **Embeddings:** Splits text into chunks and computes embeddings via OpenAI (using LangChain).
- **Database Storage:** Stores chunks and embeddings in PostgreSQL for fast retrieval.
- **RAG Query:** Retrieves the most relevant chunks via cosine similarity and sends them to ChatGPT.
- **Docker Compose:** Easily spin up the PostgreSQL database along with the app.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fast-doc-rag.git
cd fast-doc-rag
```

### 2. Configure Environment Variables

Create a `.env` file (or set these variables in your environment) with the following:

```bash
OPENAI_API_KEY=your_openai_api_key
DB_USER=fastdoc
DB_PASSWORD=fastdocpassword
DB_HOST=db
DB_PORT=5432
DB_NAME=fastdocdb
```

### 3. Docker Compose

You can launch the app and PostgreSQL using Docker Compose:

```bash
docker-compose up --build
```

This will start:
- A PostgreSQL container.
- The Flask app on port 5000.

## API Endpoints

- **POST /embed**  
  Upload a document (e.g., PDF) via form-data (key: `file`). The endpoint extracts text, computes embeddings, and inserts each chunk into PostgreSQL.

- **POST /query**  
  Send a JSON payload with a `"query"` key. The endpoint retrieves stored chunks from PostgreSQL, computes similarity, and then calls ChatGPT with the most relevant context.

## Extending the Parser

Currently, `pdf_parser.py` handles PDFs. You can extend it to support other formats (e.g., plain text) by adding new functions and updating the `/embed` endpoint accordingly.
