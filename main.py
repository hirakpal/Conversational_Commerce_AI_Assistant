from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. ADD THESE IMPORTS AT THE TOP SO PYTHON CAN FIND YOUR MODULES
from intent_engine import IntentEngine
from context_manager import ContextManager
from rag_pipeline import MissEmilyRAGPipeline

app = FastAPI(title="Miss Emily Commerce AI Clone")

# Initialize modules
intent_engine = IntentEngine()
context_manager = ContextManager()
rag_pipeline = MissEmilyRAGPipeline(vector_store_url="http://localhost:6333")

class QueryRequest(BaseModel):
    session_id: str
    user_id: str
    query: str

@app.post("/api/v1/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        # 1. Pull Context History & Intent
        history = context_manager.get_history(request.session_id)
        
        # 2. Extract Intent from modern text or follow-ups
        intent = intent_engine.extract(request.query)
        
        # 3. Simulate getting mock embedding for the category/intent
        mock_vector = [0.1] * 1536 
        
        # 4. Fetch Products (RAG Layer)
        products = rag_pipeline.retrieve_products(intent_vector=mock_vector)
        
        # 5. Fetch Mock Personalization Signals for user (e.g., from an upstream service)
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
