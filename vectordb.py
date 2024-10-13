import os
import streamlit as st
import logging
import tempfile
import shutil
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import List, Tuple, Dict, Any, Optional

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file using SentenceTransformer embeddings.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    
    # Create a temporary directory to save the file
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    
    # Write the uploaded file to the temporary path
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
    
    # Load and process the PDF using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(path)
    data = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Initialize SentenceTransformer model and HuggingFaceEmbeddings
    # sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Persist the vector database to a directory
    persist_directory = 'db'

    # Create Chroma vector database from document chunks
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding, 
        persist_directory=persist_directory
    )
    logger.info("Vector DB created and persisted")

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")

    return vector_db


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")
