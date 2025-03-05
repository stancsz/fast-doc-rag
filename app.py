import os
from flask import Flask, request, jsonify
from pdf_parser import pdf_to_text, split_text
from embeddings import compute_embeddings, retrieve_chunks
from chatgpt_api import query_chatgpt
from db import init_db, insert_document_chunks, get_all_document_chunks

# Initialize Flask app and database
app = Flask(__name__)
init_db()

# Endpoint to process a document and store embeddings into PostgreSQL
@app.route("/embed", methods=["POST"])
def embed():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file temporarily
    temp_file = "temp_uploaded_file"
    file.save(temp_file)

    # For now, we assume PDF parsing. You can extend this for other file types.
    text = pdf_to_text(temp_file)
    if not text:
        os.remove(temp_file)
        return jsonify({"error": "No text extracted from document"}), 400

    # Split text into chunks
    chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
    
    # Compute embeddings for each chunk
    embeddings, embedding_model = compute_embeddings(chunks)

    # Insert each chunk and its embedding into PostgreSQL
    insert_document_chunks(chunks, embeddings)

    os.remove(temp_file)
    return jsonify({"message": "Document processed and embeddings stored.", "num_chunks": len(chunks)})

# Endpoint to query the stored embeddings using a natural language query
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    query_text = data["query"]

    # Get all stored document chunks and embeddings from PostgreSQL
    rows = get_all_document_chunks()
    if not rows:
        return jsonify({"error": "No document data found. Please run /embed first."}), 400

    # rows is a list of tuples: (id, chunk, embedding)
    chunks = [row[1] for row in rows]
    embeddings = [row[2] for row in rows]

    # Reinitialize the embedding model for the query
    from langchain.embeddings.openai import OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings()

    # Retrieve the top matching chunks
    top_chunks = retrieve_chunks(query_text, chunks, embeddings, embedding_model, top_k=3)
    context = "\n\n".join(top_chunks)

    # Query ChatGPT with the retrieved context
    answer = query_chatgpt(query_text, context)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Run the Flask app on 0.0.0.0 so it can be accessed via Docker
    app.run(host="0.0.0.0", port=5000, debug=True)