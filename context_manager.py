import os
import redis
import json

class ContextManager:
    def __init__(self):
        # Read variables configured via Streamlit Secrets
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", 12621))
        redis_password = os.environ.get("REDIS_PASSWORD", None)

        self.db = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            password=redis_password, 
            decode_responses=True
        )
        self.ttl = 1800  # 30-minute session retention

    def save_turn(self, session_id: str, user_query: str, ai_response: str, intent_data: dict):
        session_key = f"session:{session_id}"
        history = self.get_history(session_id)
        history.append({"user": user_query, "assistant": ai_response})
        
        payload = {
            "history": history,
            "last_intent": intent_data
        }
        self.db.setex(session_key, self.ttl, json.dumps(payload))

    def get_history(self, session_id: str) -> list:
        data = self.db.get(f"session:{session_id}")
        if data:
            return json.loads(data).get("history", [])
        return []

    def get_last_intent(self, session_id: str) -> dict:
        data = self.db.get(f"session:{session_id}")
        if data:
            return json.loads(data).get("last_intent", {})
        return {}
