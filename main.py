import os
import asyncio
import uvicorn
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. IMPORTS FROM YOUR MODULES
from intent_engine import IntentEngine
from context_manager import ContextManager
from rag_pipeline import MissEmilyRAGPipeline

# 2. INITIALIZE FASTAPI Backend
app = FastAPI(title="Miss Emily Commerce AI Clone")

# Initialize core engines safely
intent_engine = IntentEngine()
context_manager = ContextManager()
rag_pipeline = MissEmilyRAGPipeline()  # Uses your Pinecone cloud integration

class QueryRequest(BaseModel):
    session_id: str
    user_id: str
    query: str

@app.post("/api/v1/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        history = context_manager.get_history(request.session_id)
        intent = intent_engine.extract(request.query)
        mock_vector = [0.1] * 1536 
        products = rag_pipeline.retrieve_products(intent_vector=mock_vector, query=request.query)
        user_signals = {"brand_affinity": "Apple", "price_sensitivity": "Mid-to-High"}
        
        ai_response = rag_pipeline.generate_response(
            query=request.query,
            history=history,
            context_products=products,
            user_signals=user_signals
        )
        
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
# 3. ASYNC BACKGROUND RUNNER FOR FASTAPI
# =====================================================================
# This snippet runs Uvicorn in the background so it doesn't freeze Streamlit!
@st.cache_resource
def start_fastapi_backend():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Run the uvicorn server inside the existing Streamlit async event loop
    loop = asyncio.get_event_loop()
    loop.create_task(server.serve())

# Spin up the background API server seamlessly
start_fastapi_backend()


# =====================================================================
# 4. STREAMLIT FRONTEND UI LAYER (Satisfies Health Checks)
# =====================================================================
st.set_page_config(page_title="Miss Emily AI Assistant", page_icon="🛍️")

st.title("🛍️ Miss Emily Commerce AI")
st.caption("Backend API running on `http://127.0.0.1:8000` | UI Online")

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_demo_session"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation UI history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle Chat Input
if user_query := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Processing request..."):
        try:
            history = context_manager.get_history(st.session_state.session_id)
            intent_data = intent_engine.extract(user_query)
            products = rag_pipeline.retrieve_products(query=user_query)
            
            ai_response = rag_pipeline.generate_response(
                query=user_query,
                history=history,
                context_products=products,
                user_signals={"brand_affinity": "All", "price_sensitivity": "Balanced"}
            )
            
            context_manager.save_turn(
                session_id=st.session_state.session_id,
                user_query=user_query,
                ai_response=ai_response,
                intent_data=intent_data.model_dump()
            )
            
            with st.chat_message("assistant"):
                st.write(ai_response)
                with st.expander("🔍 Engine Diagnostics"):
                    st.json({"Intent": intent_data.category, "Retrieved Items": products})
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            st.error(f"UI Pipeline Exception: {e}")
