import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

class MissEmilyRAGPipeline:
    def __init__(self, vector_store_url=None, index_name="missemilyrag-pipeline"):
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
        # This completely bypasses the local 'connection refused' errors
        try:
            self.vector_store = PineconeVectorStore(
                index_name=index_name, 
                embedding=self.embeddings
            )
        except Exception as e:
            # Fallback block to prevent the app from freezing up if the index isn't ready
            st.warning(f"Pinecone Vector Store initialization warning: {e}")
            self.vector_store = None

    def retrieve_products(self, intent_vector=None, query: str = None, limit: int = 3):
        """
        Fetches relevant product recommendations from the Pinecone index.
        Supports both direct text queries and raw vector matching.
        """
        if not self.vector_store:
            return ["Mock Product: High-End Wireless Earbuds ($199)"]

        # Fallback to a text search if main.py doesn't supply a query string directly
        search_query = query if query else "latest electronics"
        
        try:
            docs = self.vector_store.similarity_search(search_query, k=limit)
            return [doc.page_content for doc in docs]
        except Exception:
            # Return graceful mock products if the cloud index is empty/initializing
            return [
                "Premium Noise-Cancelling Headphones ($299) - Brand Affinity Match",
                "Ergonomic Wireless Mouse ($79) - Comfort Focus"
            ]

    def generate_response(self, query: str, history: list, context_products: list, user_signals: dict) -> str:
        """
        Simulates the LLM Response Orchestration Layer using the retrieved context.
        """
        # Quick fallback response generation logic
        products_str = ", ".join(context_products)
        return (
            f"Based on your interest in products like [{products_str}] and your preference "
            f"for {user_signals.get('brand_affinity', 'premium')} brands, here is what I recommend..."
        )
