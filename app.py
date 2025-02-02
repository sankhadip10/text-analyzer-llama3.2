# import os
# import time
# from typing import TypedDict, List
# from langgraph.graph import StateGraph, END
# from langchain.prompts import PromptTemplate
# from langchain_ollama import ChatOllama
# from langchain.schema import HumanMessage
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import gradio as gr
# import requests
# import uvicorn
# import threading
# from contextlib import contextmanager
# import socket
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # LLM Model
# llm = ChatOllama(model="llama3.2:latest", base_url="http://127.0.0.1:11434")
#
#
# # Define state format
# class State(TypedDict):
#     text: str
#     classification: str
#     entities: List[str]
#     summary: str
#
#
# # Request model for FastAPI
# class TextInput(BaseModel):
#     text: str
#
#
# # Helper function to check if port is available
# def is_port_available(port: int) -> bool:
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         try:
#             s.bind(('127.0.0.1', port))
#             return True
#         except socket.error:
#             return False
#
#
# # Wait for server to be ready
# def wait_for_server(url: str, timeout: int = 30, interval: float = 0.5):
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             requests.get(url)
#             return True
#         except requests.exceptions.RequestException:
#             time.sleep(interval)
#     return False
#
#
# # Nodes for LangGraph workflow
# def classification_node(state: State):
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText: {text}\n\nCategory:",
#     )
#     message = HumanMessage(content=prompt.format(text=state["text"]))
#     try:
#         classification = llm.invoke([message]).content.strip()
#         return {"classification": classification}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
#
#
# def entity_extraction_node(state: State):
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText: {text}\n\nEntities:"
#     )
#     message = HumanMessage(content=prompt.format(text=state["text"]))
#     try:
#         entities = llm.invoke([message]).content.strip().split(", ")
#         return {"entities": entities}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")
#
#
# def summarization_node(state: State):
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template="Summarize the following text in one short sentence.\n\nText: {text}\n\nSummary:"
#     )
#     message = HumanMessage(content=prompt.format(text=state["text"]))
#     try:
#         summary = llm.invoke([message]).content.strip()
#         return {"summary": summary}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
#
#
# # Define LangGraph workflow
# workflow = StateGraph(State)
# workflow.add_node("classification_node", classification_node)
# workflow.add_node("entity_extraction", entity_extraction_node)
# workflow.add_node("summarization", summarization_node)
#
# workflow.set_entry_point("classification_node")
# workflow.add_edge("classification_node", "entity_extraction")
# workflow.add_edge("entity_extraction", "summarization")
# workflow.add_edge("summarization", END)
#
# compiled_workflow = workflow.compile()
#
#
# @app.post("/analyze")
# async def analyze_text(input_data: TextInput):
#     if not input_data.text.strip():
#         raise HTTPException(status_code=400, detail="Text input cannot be empty")
#
#     try:
#         result = compiled_workflow.invoke({"text": input_data.text})
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# def find_available_port(start_port: int = 7860, max_attempts: int = 100) -> int:
#     for port in range(start_port, start_port + max_attempts):
#         if is_port_available(port):
#             return port
#     raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts}")
#
#
# def main():
#     try:
#         # Find available ports for both servers
#         fastapi_port = find_available_port(7860)
#         gradio_port = find_available_port(fastapi_port + 1)
#
#         # Start FastAPI server
#         fastapi_thread = threading.Thread(
#             target=lambda: uvicorn.run(app, host="127.0.0.1", port=fastapi_port),
#             daemon=True
#         )
#         fastapi_thread.start()
#
#         # Wait for FastAPI server to be ready
#         fastapi_url = f"http://127.0.0.1:{fastapi_port}"
#         if not wait_for_server(fastapi_url):
#             raise RuntimeError("FastAPI server failed to start")
#
#         print(f"FastAPI server running on {fastapi_url}")
#
#         # Gradio interface
#         def process_text(text):
#             try:
#                 response = requests.post(f"{fastapi_url}/analyze", json={"text": text})
#                 response.raise_for_status()
#                 result = response.json()
#                 return (
#                     result["classification"],
#                     ", ".join(result["entities"]),
#                     result["summary"]
#                 )
#             except Exception as e:
#                 return f"Error: {str(e)}", "", ""
#
#         interface = gr.Interface(
#             fn=process_text,
#             inputs=gr.Textbox(label="Enter text to analyze"),
#             outputs=[
#                 gr.Textbox(label="Classification"),
#                 gr.Textbox(label="Entities"),
#                 gr.Textbox(label="Summary")
#             ],
#             title="Text Analysis Application",
#             description="Enter text to get classification, entity extraction, and summarization."
#         )
#
#         # Launch Gradio interface without sharing
#         interface.launch(
#             server_port=gradio_port,
#             share=False,  # Disable sharing to avoid the frpc issue
#             server_name="127.0.0.1"  # Bind to localhost only
#         )
#
#     except Exception as e:
#         print(f"Error starting application: {str(e)}")
#         raise
#
#
# if __name__ == "__main__":
#     main()

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Text Analysis API")

# LLM Model with error handling
def initialize_llm():
    try:
        return ChatOllama(model="llama3.2:latest", base_url="http://127.0.0.1:11434")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise RuntimeError("Failed to initialize LLM. Please check if Ollama is running.")

llm = initialize_llm()

# Define state format
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

# Define LangGraph workflow
workflow = StateGraph(State)

def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText: {text}\n\nCategory:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    try:
        classification = llm.invoke([message]).content.strip()
        logger.debug(f"Classification result: {classification}")
        return {"classification": classification}
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText: {text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    try:
        entities = llm.invoke([message]).content.strip().split(", ")
        logger.debug(f"Extracted entities: {entities}")
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")

def summarization_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText: {text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    try:
        summary = llm.invoke([message]).content.strip()
        logger.debug(f"Generated summary: {summary}")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

# Set up workflow
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

compiled_workflow = workflow.compile()

def process_text(text: str) -> tuple[str, str, str]:
    """Process text through the workflow with enhanced error handling"""
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return "Error: Empty text not allowed", "", ""

        if len(text.strip()) > 5000:  # Add reasonable limit
            logger.warning("Text too long")
            return "Error: Text exceeds 5000 characters", "", ""

        result = compiled_workflow.invoke({"text": text})
        return (
            result["classification"],
            ", ".join(result["entities"]),
            result["summary"]
        )
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return f"Error: {str(e)}", "", ""

# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            label="Enter text to analyze",
            placeholder="Type or paste your text here...",
            lines=5
        )
    ],
    outputs=[
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Entities"),
        gr.Textbox(label="Summary")
    ],
    title="Text Analysis Application",
    description="Enter text to get classification, entity extraction, and summarization.",
    examples=[
        ["The new iPhone 15 Pro was released yesterday by Apple in Cupertino. CEO Tim Cook presented the device."],
        ["A recent study by researchers at MIT shows promising results in renewable energy efficiency."],
        ["This is a blog post about my recent trip to Paris. The Eiffel Tower was amazing!"]
    ],
    theme=gr.themes.Soft()
)

# For Hugging Face Spaces, we need to use their specific configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

