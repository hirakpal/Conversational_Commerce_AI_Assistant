import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. IMPORTS FROM YOUR MODULES
from intent_engine import IntentEngine
from context_manager import ContextManager
from rag_pipeline import MissEmilyRAGPipeline

# 2. INITIALIZE FASTAPI (For backend API usage)
app = FastAPI(title="Miss Emily Commerce AI Clone")

# Initialize core engines
intent_engine = IntentEngine()
context_manager = ContextManager()
rag_pipeline = MissEmilyRAGPipeline()  # Cleaned up to use your new Pinecone backend

class QueryRequest(BaseModel):
    session_id: str
    user_id: str
    query: str

@app.post("/api/v1/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        # 1. Pull Context History & Intent
        history = context_manager.get_history(request.session_id)
        
        # 2. Extract Intent
        intent = intent_engine.extract(request.query)
        
        # 3. Simulate getting mock embedding 
        mock_vector = [0.1] * 1536 
        
        # 4. Fetch Products (RAG Layer)
        products = rag_pipeline.retrieve_products(intent_vector=mock_vector, query=request.query)
        
        # 5. Fetch Mock Personalization Signals
        user_signals = {"brand_affinity": "Apple", "price_sensitivity": "Mid-to-High"}
        
        # 6. Run LLM Response Orchestration Layer
        ai_response = rag_pipeline.generate_response(
            query=request.query,
            history=history,
            context_products=products,
            user_signals=user_signals
        )
        
        # 7. Update conversation memory state
        context_manager.save_turn(
            session_id=request.session_id,
            user_query=request.query,
            ai_response=ai_response,
            intent_data=intent.model_dump()
        )
        
        return {
            "response": ai_response,
            "extracted_intent": intent,
            "suggested_actions": ["Compare specs", "Check dimension compatibility"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# 3. STREAMLIT FRONTEND UI LAYER (Solves the health check freeze)
# =====================================================================
st.set_page_config(page_title="Miss Emily AI Assistant", page_icon="🛍️", layout="centered")

st.title("🛍️ Miss Emily Commerce AI")
st.caption("Powered by FastAPI, LangChain, Pinecone & Redis")

# Simple Session Management for the UI demo
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_demo_session"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history from local state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Chat Input
if user_query := st.chat_input("Ask me about products, compatibility, or specifications..."):
    # Display user query
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Process through the live backend engine pipeline
    with st.spinner("Processing request..."):
        try:
            # 1. Fetch historical data from Redis
            history = context_manager.get_history(st.session_state.session_id)
            
            # 2. Run intent extraction
            intent_data = intent_engine.extract(user_query)
            
            # 3. Query the Pinecone RAG database
            products = rag_pipeline.retrieve_products(query=user_query)
            
            # 4. Generate the optimized final text recommendation
            ai_response = rag_pipeline.generate_response(
                query=user_query,
                history=history,
                context_products=products,
                user_signals={"brand_affinity": "All", "price_sensitivity": "Balanced"}
            )
            
            # 5. Commit turn to Redis
            context_manager.save_turn(
                session_id=st.session_state.session_id,
                user_query=user_query,
                ai_response=ai_response,
                intent_data=intent_data.model_dump()
            )
            
            # Render response
            with st.chat_message("assistant"):
                st.write(ai_response)
                with st.expander("🔍 Behind the Scenes: Extracted Engine Data"):
                    st.json({
                        "Intent Category": intent_data.category,
                        "Intent Type": intent_data.intent_type,
                        "Extracted Constraints": intent_data.constraints,
                        "Context Retrieved": products
                    })
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            st.error(f"Engine connection pipeline error: {e}")
