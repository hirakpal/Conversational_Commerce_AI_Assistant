from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
os.environ['OPENAI_API_KEY'] = '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'

# Define structured output matching Rufus specifications
class ShoppingIntent(BaseModel):
    category: str = Field(description="The core product category being discussed.")
    use_case: Optional[str] = Field(None, description="The intended scenario or environment (e.g., travel, small apartment).")
    constraints: List[str] = Field(default=[], description="Extracted preferences like 'lightweight', 'compact', or budget limits.")
    intent_type: str = Field(description="Type: 'Discovery', 'Comparison', 'Compatibility', or 'Education'")

class IntentEngine:
    def __init__(self):
        # Using a highly logical model for accurate schema extraction
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ShoppingIntent)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Intent Extraction Engine of an advanced commerce AI. Analyze user text and output strict JSON matching the schema."),
            ("user", "{query}")
        ])
        self.chain = self.prompt | self.llm

    def extract(self, query: str) -> ShoppingIntent:
        return self.chain.invoke({"query": query})
