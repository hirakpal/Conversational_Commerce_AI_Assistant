import os
import streamlit as st  
import redis
import json

class ContextManager:
    def __init__(self):
        # 1. Pull the raw variables from your Streamlit Secrets panel
        redis_host = st.secrets.get("REDIS_HOST", "localhost")
        redis_port = st.secrets.get("REDIS_PORT", 12621)
        redis_password = st.secrets.get("REDIS_PASSWORD", "")

        try:
            # 2. Build a bulletproof Redis connection URL string
            # Format: redis://:password@host:port
            redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
            
            # 3. Connect using the explicit URL configuration parser
            self.db = redis.Redis.from_url(redis_url, decode_responses=True)
            
            # Quick ping test to fail early if credentials are wrong
            self.db.ping()
        except Exception as e:
            # Graceful internal fallback if credentials completely mismatch
            print(f"Redis initialization failed: {e}")
            self.db = None

        self.ttl = 1800  # 30-minute session retention

    def save_turn(self, session_id: str, user_query: str, ai_response: str, intent_data: dict):
        if not self.db:
            return
        session_key = f"session:{session_id}"
        history = self.get_history(session_id)
        history.append({"user": user_query, "assistant": ai_response})
        
        payload = {
            "history": history,
            "last_intent": intent_data
        }
        self.db.setex(session_key, self.ttl, json.dumps(payload))

    def get_history(self, session_id: str) -> list:
        if not self.db:
            return []
        try:
            data = self.db.get(f"session:{session_id}")
            if data:
                return json.loads(data).get("history", [])
        except Exception:
            pass
        return []

    def get_last_intent(self, session_id: str) -> dict:
        if not self.db:
            return {}
        try:
            data = self.db.get(f"session:{session_id}")
            if data:
                return json.loads(data).get("last_intent", {})
        except Exception:
            pass
        return {}
