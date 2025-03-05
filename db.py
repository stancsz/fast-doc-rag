import os
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION

# Read database connection parameters from environment variables
DB_USER = os.getenv("DB_USER", "fastdoc")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fastdocpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fastdocdb")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    id = Column(Integer, primary_key=True, index=True)
    chunk = Column(Text, nullable=False)
    # Embedding stored as an array of floats
    embedding = Column(ARRAY(DOUBLE_PRECISION), nullable=False)

def init_db():
    Base.metadata.create_all(bind=engine)

def insert_document_chunks(chunks, embeddings):
    """
    Insert chunks and their embeddings into the database.
    `embeddings` should be a NumPy array; we convert each to a list.
    """
    session = SessionLocal()
    try:
        for chunk, emb in zip(chunks, embeddings):
            doc_chunk = DocumentChunk(chunk=chunk, embedding=emb.tolist())
            session.add(doc_chunk)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_all_document_chunks():
    """
    Retrieve all document chunks and embeddings from the database.
    Returns a list of tuples: (id, chunk, embedding).
    """
    session = SessionLocal()
    try:
        rows = session.query(DocumentChunk).all()
        return [(row.id, row.chunk, row.embedding) for row in rows]
    finally:
        session.close()