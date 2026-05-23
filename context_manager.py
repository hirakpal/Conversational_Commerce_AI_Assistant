import redis
import json

class ContextManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.db = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ttl = 1800  # 30-minute session retention

    def save_turn(self, session_id: str, user_query: str, ai_response: str, intent_data: dict):
        session_key = f"session:{session_id}"
        
        # Pull current history
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
