"""
Embeddings Module for Healthcare AI Assistant.

Handles initialization of the HuggingFaceEmbeddings model, ChromaDB client,
document loading, chunking, and the full ingestion pipeline.
"""

import os
import logging
from pathlib import Path
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings

logger = logging.getLogger(__name__)

def get_embedding_function() -> HuggingFaceEmbeddings:
    """Initialize and return the HuggingFace embedding function."""
    settings = get_settings()
    logger.info("Initializing HuggingFaceEmbeddings with model: %s", settings.EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_chroma_client() -> chromadb.PersistentClient:
    """Initialize and return a persistent ChromaDB client."""
    settings = get_settings()
    logger.info("Initializing persistent ChromaDB client at: %s", settings.CHROMA_PERSIST_DIR)
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

def get_or_create_collection(client: chromadb.PersistentClient, embedding_function) -> Chroma:
    """Get or create the Chroma collection and return the LangChain vectorstore object."""
    settings = get_settings()
    return Chroma(
        client=client,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function
    )

def load_documents_from_folder(folder_path: str) -> list:
    """Load text documents from the given folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    
    logger.info("Loading documents from %s", folder_path)
    loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError("No documents found in data directory")
        
    logger.info("Loaded %d documents", len(documents))
    return documents

def split_documents(documents: list) -> list:
    """Split documents into smaller chunks for vector embedding."""
    settings = get_settings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for chunk in chunks:
        source_path = chunk.metadata.get("source", "")
        if source_path:
            chunk.metadata["source"] = os.path.basename(source_path)
            
    logger.info("Created %d chunks", len(chunks))
    return chunks

def ingest_documents(reset_collection: bool = False) -> dict:
    """Orchestrates the full document ingestion pipeline."""
    settings = get_settings()
    try:
        embedding_fn = get_embedding_function()
        client = get_chroma_client()
        
        if reset_collection:
            logger.info("Resetting collection: %s", settings.CHROMA_COLLECTION_NAME)
            try:
                client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            except Exception as e:
                logger.warning("Could not delete collection %s (it might not exist yet): %s", settings.CHROMA_COLLECTION_NAME, e)
                
        vectorstore = get_or_create_collection(client, embedding_fn)
        
        documents = load_documents_from_folder(settings.DATA_DIR)
        chunks = split_documents(documents)
        
        logger.info("Adding chunks to vectorstore")
        vectorstore.add_documents(chunks)
        
        logger.info("Ingestion complete")
        return {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "message": f"Successfully ingested {len(documents)} documents into {len(chunks)} chunks"
        }
    except Exception as e:
        logger.error("Error occurred during document ingestion: %s", str(e), exc_info=True)
        raise

