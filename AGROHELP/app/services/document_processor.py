import os
from typing import List, Dict, Any, Optional
import logging

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class DocumentProcessor:
    """
    Handles the processing of agricultural documents for the RAG pipeline.
    """
    
    def __init__(self, embedding_type: str = "huggingface", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            embedding_type: Type of embedding model to use ("huggingface" or "openai")
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        if embedding_type == "huggingface":
            # Use a multilingual model to better support Indian languages
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
        else:
            raise ValueError(f"Embedding type {embedding_type} not supported")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        logging.info(f"Initialized DocumentProcessor with {embedding_type} embeddings")
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process a PDF file and return a list of document chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return []
        
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            logging.info(f"Processed PDF {file_path}: {len(split_docs)} chunks created")
            return split_docs
            
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def create_vector_store(self, documents: List[Document], store_type: str = "faiss") -> Any:
        """
        Create a vector store from a list of documents.
        
        Args:
            documents: List of document chunks
            store_type: Type of vector store to create ("faiss" or "chroma")
            
        Returns:
            Vector store object
        """
        if not documents:
            logging.warning("No documents provided for vector store creation")
            return None
        
        try:
            if store_type == "faiss":
                vector_store = FAISS.from_documents(documents, self.embeddings)
                logging.info(f"Created FAISS vector store with {len(documents)} documents")
                return vector_store
            
            elif store_type == "chroma":
                vector_store = Chroma.from_documents(documents, self.embeddings)
                logging.info(f"Created Chroma vector store with {len(documents)} documents")
                return vector_store
            
            else:
                logging.error(f"Unsupported vector store type: {store_type}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            return None
    
    def search_documents(self, vector_store: Any, query: str, k: int = 5) -> List[Document]:
        """
        Search for relevant documents in the vector store.
        
        Args:
            vector_store: Vector store object
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if vector_store is None:
            logging.warning("No vector store provided for search")
            return []
        
        try:
            docs = vector_store.similarity_search(query, k=k)
            logging.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
            
        except Exception as e:
            logging.error(f"Error searching documents: {str(e)}")
            return []
