from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI

class RufusRAGPipeline:
    def __init__(self, vector_store_url: str):
        self.qdrant = QdrantClient(url=vector_store_url)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    def retrieve_products(self, intent_vector: list, limit: int = 3) -> list:
        # Vector similarity search across catalog data and product review insights
        results = self.qdrant.search(
            collection_name="commerce_catalog",
            query_vector=intent_vector,
            limit=limit
        )
        return [res.payload for res in results]

    def generate_response(self, query: str, history: list, context_products: list, user_signals: dict) -> str:
        # Inject context, behavioral signals, and factual payload to eliminate hallucinations
        product_context_str = json.dumps(context_products, indent=2)
        
        system_prompt = f"""
        You are Miss Emily, an elite generative AI shopping assistant.
        Your goal is to guide the user to the best purchase decision seamlessly.

        CRITICAL GROUNDING DATA (Strictly adhere to this context to prevent hallucinations):
        {product_context_str}

        USER PERSONALIZATION PROFILE:
        - Brand Affinity: {user_signals.get('brand_affinity', 'None')}
        - Price Sensitivity: {user_signals.get('price_sensitivity', 'Mid-range')}
        
        Formulate a natural, conversational response. Highlight key specifications, address customer review insights if requested, and offer dynamic guidance.
        """
        
        messages = [("system", system_prompt)]
        for turn in history[-3:]: # Inject past multi-turn context
            messages.append(("user", turn["user"]))
            messages.append(("assistant", turn["assistant"]))
        messages.append(("user", query))
        
        response = self.llm.invoke(messages)
        return response.content
