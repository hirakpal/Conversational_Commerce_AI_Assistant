import os
import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Define structured output matching Miss Emily specifications
class ShoppingIntent(BaseModel):
    category: str = Field(description="The core product category being discussed.")
    use_case: Optional[str] = Field(None, description="The intended scenario or environment (e.g., travel, small apartment).")
    constraints: List[str] = Field(default=[], description="Extracted preferences like 'lightweight', 'compact', or budget limits.")
    intent_type: str = Field(description="Type: 'Discovery', 'Comparison', 'Compatibility', or 'Education'")

class IntentEngine:
    def __init__(self):
        # 1. Safely load the key from Streamlit secrets into the runtime environment
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        
        # 2. Initialize the model securely
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ShoppingIntent)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Intent Extraction Engine of an advanced commerce AI. Analyze user text and output strict JSON matching the schema."),
            ("user", "{query}")
        ])
        self.chain = self.prompt | self.llm

    def extract(self, query: str) -> ShoppingIntent:
        return self.chain.invoke({"query": query})
