#code for running in local server
# import gradio as gr
# import requests
#
# # Backend API URL
# API_URL = "http://localhost:5000/analyze"
#
#
# def process_text(text):
#     """Send text to backend API and retrieve classification, entities, and summary."""
#     response = requests.post(API_URL, json={"text": text})
#
#     if response.status_code == 200:
#         result = response.json()
#         classification = result.get("classification", "N/A")
#         entities = result.get("entities", [])
#         summary = result.get("summary", "N/A")
#         return classification, ", ".join(entities), summary
#     else:
#         return "Error", "Error", "Error"
#
#
# # Gradio UI
# interface = gr.Interface(
#     fn=process_text,
#     inputs=gr.Textbox(label="Enter text"),
#     outputs=[
#         gr.Textbox(label="Classification"),
#         gr.Textbox(label="Entities"),
#         gr.Textbox(label="Summary"),
#     ],
#     title="Text Analysis App",
#     description="Analyze text using Llama 3.2: Classify, Extract Entities, and Summarize.",
# )
#
# if __name__ == "__main__":
#     interface.launch()
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import requests
import uvicorn
import threading

# Initialize FastAPI app
app = FastAPI()

# LLM Model
llm = ChatOllama(model="llama3.2:latest", base_url="http://127.0.0.1:11434")

# Define state format
class State(TypedDict):
    text: str
    classification: List[str]
    entities: List[str]
    summary: str

# Request model for FastAPI
class TextInput(BaseModel):
    text: str

# Nodes for LangGraph workflow
def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories:News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

def summarization_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

# Define LangGraph workflow
workflow = StateGraph(State)
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

compiled_workflow = workflow.compile()

@app.post("/analyze")
def analyze_text(input_data: TextInput):
    state_input = {"text": input_data.text}
    result = compiled_workflow.invoke(state_input)
    return result

# Start FastAPI server in a thread
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=7860)

threading.Thread(target=run_server, daemon=True).start()

# Gradio UI
def process_text(text):
    response = requests.post("http://localhost:7860/analyze", json={"text": text})
    result = response.json()
    return result["classification"], ", ".join(result["entities"]), result["summary"]

interface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(label="Enter text"),
    outputs=[
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Entities"),
        gr.Textbox(label="Summary"),
    ],
    title="Text Analysis App",
)

interface.launch(share=True)


