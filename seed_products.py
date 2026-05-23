import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

def seed_database():
    print("🚀 Initializing connection to Pinecone cloud index...")
    
    # 1. Direct configuration fallbacks (Use your actual keys if running locally)
    os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "your-openai-key-here")
    os.environ["PINECONE_API_KEY"] = st.secrets.get("PINECONE_API_KEY", "your-pinecone-key-here")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = "missemilyrag-pipeline"

    # 2. Define premium sample product data to upload
    sample_products = [
        Document(
            page_content="iPhone 17 Pro Max (256GB, Titanium Gray, A20 Chip) - $1,199. Premium Apple smartphone with advanced AI integration.",
            metadata={"category": "Electronics", "brand": "Apple"}
        ),
        Document(
            page_content="ASUS ROG Strix G16 Gaming Laptop (Intel i9, NVIDIA RTX 4080, 32GB DDR5 RAM, 1TB SSD) - $2,199. High-end computing powerhouse.",
            metadata={"category": "Laptops", "brand": "ASUS"}
        ),
        Document(
            page_content="Sony WH-1000XM5 Wireless Headphones (Active Noise Cancelling, 30hr Battery Life, Silver) - $349. Top tier premium audio.",
            metadata={"category": "Electronics", "brand": "Sony"}
        ),
         ),
        Document(
            page_content="Samsung Galaxy S26 Ultra (512GB, Phantom Black) - $1,299. Flagship smartphone with 200MP camera and built-in stylus.",
            metadata={"category": "Electronics", "brand": "Samsung"}
        ),
        Document(
            page_content="Apple MacBook Pro 14-inch (M4 Pro Chip, 24GB Unified Memory, 512GB SSD) - $1,999. Professional creator laptop.",
            metadata={"category": "Laptops", "brand": "Apple"}
        )
    ]

    print(f"📦 Seeding {len(sample_products)} items to index '{index_name}'...")
    try:
        # 3. Stream data up to your Pinecone vector store cluster
        vector_store = PineconeVectorStore.from_documents(
            documents=sample_products,
            embedding=embeddings,
            index_name=index_name
        )
        print("✅ Database successfully seeded! Your items are now live.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    seed_database()
