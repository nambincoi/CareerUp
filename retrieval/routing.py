import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Access environment variables
tracing_enabled = os.getenv('LANGCHAIN_TRACING_V2')
endpoint = os.getenv('LANGCHAIN_ENDPOINT')
api_key = os.getenv('LANGCHAIN_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Function to dynamically get the list of files in a folder
def get_datasource_list(folder_path):
    """Fetch the list of filenames (without extension) from the specified folder."""
    try:
        files = os.listdir(folder_path)
        # Remove extensions and filter only relevant files (e.g., .pdf, .json, .txt)
        datasources = [f for f in files]
        return datasources
    except FileNotFoundError:
        print(f"[ERROR] The folder '{folder_path}' was not found.")
        return []

# Fetch the list of files from your folder
datasource_list = get_datasource_list('/home/hoangnam/rag/rag-from-scratch/vietnam_university_rag/retrieval')
print(datasource_list)

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: str = Field(
        ...,
        description=f"Given a user question, choose which datasource from the following list would be most relevant for answering their question: {datasource_list}",
    )

# LLM with function call 
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt 
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router 
router = prompt | structured_llm

question = """Tôi muốn tìm hiểu về đề án trường đại học bách khoa hà nội.
"""

result = router.invoke({"question": question})
result.datasource

from typing import List
from typing import Literal

from cohere import SystemChatMessageV2
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import json

def route_to_datasource(question: str, datasource_list: List[str]) -> str:
    # 1) Dynamically define the Pydantic model with your datasource_list in the description
    
    class RouteQuery(BaseModel):
        datasource: str = Field(
            ...,
            description=(
                "Given a user question, choose which datasource from the following list "
                f"would be most relevant: {datasource_list}"
            ),
        )

    # 2) Build the LLM + structured‐output parser
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # 3) Create the routing prompt
    system_msg = SystemChatMessageV2(
        content=(
            "You are an expert at routing a user question to the appropriate data source.\n\n"
            "Based on the programming language or domain the question refers to, "
            "choose the single best datasource from the provided list."
        )
    )

    # 4) Invoke the chain
    #    The operator ‘|’ wires prompt → structured_llm under the hood
    prompt = ChatPromptTemplate.from_messages([("system", system_msg.content),
                                               ("human", "{question}")])
    router = prompt | structured_llm
    result = router.invoke({"question": question})

    # 5) Extract and return the chosen datasource
    return result.datasource


# --- Example usage ---
if __name__ == "__main__":
    sources = datasource_list
    q = "Tôi muốn tìm hiểu về đề án trường đại học bách khoa hà nội."
    chosen = route_to_datasource(q, sources)
    print("Routed to datasource:", chosen)
