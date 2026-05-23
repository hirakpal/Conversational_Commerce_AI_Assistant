import streamlit as st

# 1. IMPORTS FROM YOUR MODULES
from intent_engine import IntentEngine
from context_manager import ContextManager
from rag_pipeline import MissEmilyRAGPipeline

# 2. INITIALIZE ENGINES SAFELY VIA STREAMLIT CACHING
@st.cache_resource
def get_engines():
    intent_engine = IntentEngine()
    context_manager = ContextManager()
    rag_pipeline = MissEmilyRAGPipeline()
    return intent_engine, context_manager, rag_pipeline

intent_engine, context_manager, rag_pipeline = get_engines()

# 3. STREAMLIT FRONTEND UI LAYER
st.set_page_config(page_title="Miss Emily AI Assistant", page_icon="🛍️")

st.title("🛍️ Miss Emily Commerce AI")
st.caption("Powered by MissEmily RAG Pipeline & Live Context Engine")

# Setup Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_demo_session"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation UI history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle Chat Input
if user_query := st.chat_input("Ask me anything about product availability or specs..."):
    # Display user query instantly
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Thinking..."):
        try:
            # 1. Fetch historical data from Redis
            history = context_manager.get_history(st.session_state.session_id)
            
            # 2. Run intent extraction
            intent_data = intent_engine.extract(user_query)
            
            # 3. Query the Pinecone cloud vector database
            products = rag_pipeline.retrieve_products(query=user_query)
            
            # 4. Generate response
            ai_response = rag_pipeline.generate_response(
                query=user_query,
                history=history,
                context_products=products,
                user_signals={"brand_affinity": "All", "price_sensitivity": "Balanced"}
            )
            
            # 5. Commit state turn to Redis
            context_manager.save_turn(
                session_id=st.session_state.session_id,
                user_query=user_query,
                ai_response=ai_response,
                intent_data=intent_data.model_dump()
            )
            
            # Render response to the user
            with st.chat_message("assistant"):
                st.write(ai_response)
                with st.expander("🔍 Engine Diagnostics"):
                    st.json({
                        "Category": intent_data.category,
                        "Type": intent_data.intent_type,
                        "Constraints": intent_data.constraints,
                        "Retrieved Items": products
                    })
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            st.error(f"UI Pipeline Exception: {e}")
