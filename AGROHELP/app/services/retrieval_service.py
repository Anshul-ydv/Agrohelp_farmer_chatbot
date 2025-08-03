import logging
from typing import List, Dict, Any, Optional

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

class RetrievalService:
    """
    Service for retrieving information from vector stores using LLMs.
    """
    
    def __init__(self, llm: Any, vector_store: Any = None):
        """
        Initialize the retrieval service.
        
        Args:
            llm: Language model to use for retrieval
            vector_store: Vector store to retrieve from
        """
        self.llm = llm
        self.vector_store = vector_store
        self.qa_chain = None
        
        if self.vector_store:
            self._setup_qa_chain()
    
    def set_vector_store(self, vector_store: Any) -> None:
        """
        Set or update the vector store.
        
        Args:
            vector_store: Vector store to retrieve from
        """
        self.vector_store = vector_store
        self._setup_qa_chain()
    
    def _setup_qa_chain(self) -> None:
        """
        Set up the QA chain with the current vector store and LLM.
        """
        if not self.vector_store or not self.llm:
            logging.warning("Cannot set up QA chain: missing vector store or LLM")
            return
        
        # Create a custom prompt template for agricultural queries
        template = """
        You are AgriSaarthi, an AI assistant for farmers focused on sustainable agriculture.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always provide practical, actionable advice that farmers can implement.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logging.info("QA chain set up successfully")
    
    def retrieve_answer(self, query: str) -> Dict[str, Any]:
        """
        Retrieve an answer for a query using the QA chain.
        
        Args:
            query: Query to answer
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if not self.qa_chain:
            logging.warning("QA chain not set up, cannot retrieve answer")
            return {
                "answer": "I don't have access to agricultural information yet. Please upload some documents first.",
                "sources": []
            }
        
        try:
            # Run the QA chain
            result = self.qa_chain({"query": query})
            
            # Extract answer and source documents
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Format source information
            sources = []
            for doc in source_docs:
                source = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source)
            
            logging.info(f"Retrieved answer for query: {query}")
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logging.error(f"Error retrieving answer: {str(e)}")
            return {
                "answer": "I encountered an error while trying to answer your question. Please try again.",
                "sources": []
            }
    
    def generate_recommendations(self, topic: str, context: Optional[str] = None) -> str:
        """
        Generate agricultural recommendations on a specific topic.
        
        Args:
            topic: Topic to generate recommendations for
            context: Additional context for the recommendations
            
        Returns:
            Generated recommendations
        """
        if not self.llm:
            logging.warning("LLM not set up, cannot generate recommendations")
            return "I can't generate recommendations at the moment."
        
        try:
            # Create a prompt for generating recommendations
            prompt_text = f"""
            You are AgriSaarthi, an AI assistant for farmers focused on sustainable agriculture.
            Generate detailed, practical recommendations for the following agricultural topic:
            
            Topic: {topic}
            
            """
            
            if context:
                prompt_text += f"""
                Additional context:
                {context}
                """
            
            prompt_text += """
            Provide step-by-step instructions that are easy to follow.
            Include both traditional and modern sustainable approaches.
            Focus on techniques that are accessible to small-scale farmers.
            """
            
            # Generate recommendations
            response = self.llm(prompt_text)
            
            logging.info(f"Generated recommendations for topic: {topic}")
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            return "I encountered an error while trying to generate recommendations. Please try again."
