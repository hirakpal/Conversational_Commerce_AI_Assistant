import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

class MissEmilyRAGPipeline:
    def __init__(self, vector_store_url=None, index_name="emily-commerce-ai"):
        """
        Initializes the Pinecone RAG Pipeline.
        Note: vector_store_url is kept in the signature to prevent breaking main.py,
        but we safely pull configurations from Streamlit Secrets in production.
        """
        # 1. Map Streamlit Cloud Secrets safely to environment variables for LangChain
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        if "PINECONE_API_KEY" in st.secrets:
            os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

        # 2. Initialize the text embedding model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 3. Connect to your hosted Pinecone cloud vector database index
        try:
            self.vector_store = PineconeVectorStore(
                index_name=index_name, 
                embedding=self.embeddings
            )
        except Exception as e:
            st.warning(f"Pinecone Vector Store initialization warning: {e}")
            self.vector_store = None

    def retrieve_products(self, intent_vector=None, query: str = None, limit: int = 3):
        """
        Fetches relevant product recommendations from the Pinecone index.
        """
        if not self.vector_store:
            return ["Mock Product: High-End Wireless Earbuds ($199)"]

        try:
            # If main.py sends a raw text query string, convert to vectors implicitly
            if query and not intent_vector:
                docs = self.vector_store.similarity_search(query, k=limit)
            else:
                docs = self.vector_store.similarity_search_by_vector(intent_vector, k=limit)
                
            return [doc.page_content for doc in docs]
            
        except Exception as e:
            # Clean empty fallback gracefully handled by your UI
            return []

    def generate_response(self, query: str, history: list, context_products: list, user_signals: dict) -> str:
        """
        Simulates the LLM Response Orchestration Layer using the retrieved context.
        """
        products_str = ", ".join(context_products) if context_products else "matching items"
        return (
            f"Based on your interest in products like [{products_str}] and your preference "
            f"for {user_signals.get('brand_affinity', 'premium')} brands, here is what I recommend..."
        )
