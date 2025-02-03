# Text Analysis Application

This repository contains a Python-based text analysis application that leverages a LangGraph workflow to classify text, extract entities, and generate summaries. The application is built using FastAPI for the backend and Gradio for the user interface. It uses the Ollama language model (`llama3.2`) for text processing.

## Features
- **Text Classification**: Classifies input text into categories such as News, Blog, Research, or Other.
- **Entity Extraction**: Extracts entities (Person, Organization, Location) from the text.
- **Text Summarization**: Generates a concise summary of the input text.
- **User-Friendly Interface**: Provides a Gradio-based web interface for easy interaction.

## Prerequisites
Before running the application, ensure you have the following installed:
- Python 3.8 or higher
- [Ollama](https://github.com/jmorganca/ollama) running locally with the `llama3.2` model
- Required Python packages (install via `pip install -r requirements.txt`)

## Installation and Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository/text-analysis.git
   cd text-analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Ollama Locally**:
   Ensure Ollama is running with the `llama3.2` model. You can start Ollama using:
   ```bash
   ollama serve
   ```

4. **Run the Application**:
   Start the application by running:
   ```bash
   python main.py
   ```

   The application will automatically find available ports for the FastAPI backend and Gradio frontend. Once started, it will print the URLs for both servers.

5. **Access the Application**:
   Open your browser and navigate to the Gradio interface URL (e.g., `http://127.0.0.1:7861`). Enter text in the provided textbox and click "Submit" to see the classification, extracted entities, and summary.

## How It Works
1. **Text Input**: The user inputs text into the Gradio interface.
2. **FastAPI Backend**: The text is sent to the FastAPI backend, which processes it using a LangGraph workflow.
3. **LangGraph Workflow**:
   - **Classification**: The text is classified into one of the predefined categories.
   - **Entity Extraction**: Entities (Person, Organization, Location) are extracted from the text.
   - **Summarization**: A concise summary of the text is generated.
4. **Output**: The results (classification, entities, and summary) are displayed in the Gradio interface.

## Benefits
- **Automated Text Analysis**: Quickly analyze and extract key information from text.
- **Customizable Workflow**: The LangGraph workflow can be extended or modified to include additional text processing steps.
- **Local Deployment**: The application runs entirely on your local machine, ensuring data privacy and security.

## Example Output
For the input text:
```
"Elon Musk announced that Tesla will build a new factory in Texas. The factory is expected to create thousands of jobs."
```

The application might output:
- **Classification**: News
- **Entities**: Elon Musk, Tesla, Texas
- **Summary**: Tesla plans to build a new factory in Texas, creating thousands of jobs.

## Troubleshooting
- **Port Conflicts**: If the application fails to start due to port conflicts, ensure no other services are using ports 7860 and above.
- **Ollama Issues**: Ensure Ollama is running and the `llama3.2` model is available. You can check by visiting `http://127.0.0.1:11434`.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


# Text Analysis Application on Hugging Face Spaces

This repository contains a modified version of the Text Analysis Application designed to run on Hugging Face Spaces. The application uses Hugging Face's `transformers` library with the `google/flan-t5-base` model for text classification, entity extraction, and summarization. It is built using FastAPI for the backend and Gradio for the user interface.

## Live Running Space
You can access the live running application on Hugging Face Spaces here:  
[Text Analysis Application on Hugging Face Spaces](https://huggingface.co/spaces/sankhadip10/textanalyzer)

---

## Why the Original Code Won't Work on Hugging Face Spaces

The original code was designed to run locally using the Ollama language model (`llama3.2`). However, Hugging Face Spaces has specific limitations and requirements that necessitate modifications:

1. **Model Availability**:
   - The original code relies on the Ollama model (`llama3.2`), which is not available on Hugging Face Spaces. Instead, we use Hugging Face's `google/flan-t5-base` model, which is lightweight and compatible with the free tier of Spaces.

2. **Resource Constraints**:
   - Hugging Face Spaces has limited computational resources, especially on the free tier. The `flan-t5-base` model is more resource-efficient compared to larger models like `llama3.2`.

3. **Port and Server Configuration**:
   - Hugging Face Spaces does not allow custom server configurations like FastAPI or custom ports. Instead, Gradio is used directly to handle the interface and backend logic.

4. **Environment Variables**:
   - The `.env` file used in the original code is not supported on Hugging Face Spaces. Instead, environment variables are managed through the Spaces interface.

---

## Modified Code for Hugging Face Spaces

The modified code replaces the Ollama model with Hugging Face's `flan-t5-base` model and simplifies the backend to work within the constraints of Hugging Face Spaces. Here's the updated code:

```python
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from transformers import pipeline
from fastapi import FastAPI, HTTPException
import gradio as gr
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            do_sample=True,  # Added this parameter
            temperature=0.5,
            max_length=512
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

# Define state format
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

# Define LangGraph workflow
workflow = StateGraph(State)

async def classification_node(state: State):
    prompt = f"Classify as News, Blog, Research, or Other: {state['text']}\n\nClassification:"
    try:
        result = llm(prompt, max_length=50)[0]
        classification = result['generated_text'].strip()
        logger.debug(f"Classification result: {classification}")
        return {"classification": classification}
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

async def entity_extraction_node(state: State):
    prompt = f"Extract named entities (Person, Organization, Location) from: {state['text']}\n\nEntities:"
    try:
        result = llm(prompt, max_length=100)[0]
        entities = [e.strip() for e in result['generated_text'].strip().split(",") if e.strip()]
        logger.debug(f"Extracted entities: {entities}")
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")

async def summarization_node(state: State):
    prompt = f"Summarize in one sentence: {state['text']}\n\nSummary:"
    try:
        result = llm(prompt, max_length=100)[0]
        summary = result['generated_text'].strip()
        logger.debug(f"Generated summary: {summary}")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

# Initialize the LLM
llm = initialize_llm()

# Set up workflow
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

compiled_workflow = workflow.compile()

async def process_text(text: str) -> tuple[str, str, str]:
    """Process text through the workflow with enhanced error handling"""
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return "Error: Empty text not allowed", "", ""

        if len(text.strip()) > 2000:  # Reduced limit for free tier
            logger.warning("Text too long")
            return "Error: Text exceeds 2000 characters", "", ""

        result = await compiled_workflow.ainvoke({"text": text})
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
    description="Enter text to get classification, entity extraction, and summarization using Hugging Face's FLAN-T5 model.",
    examples=[
        ["The new iPhone 15 Pro was released yesterday by Apple in Cupertino. CEO Tim Cook presented the device."],
        ["A recent study by researchers at MIT shows promising results in renewable energy efficiency."],
        ["This is a blog post about my recent trip to Paris. The Eiffel Tower was amazing!"]
    ],
    theme=gr.themes.Soft()
)

# For Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

---

## Setup and Running on Hugging Face Spaces

1. **Create a Hugging Face Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
   - Choose "Gradio" as the SDK.

2. **Upload the Code**:
   - Copy the modified code into a `app.py` file in your Space repository.

3. **Add Requirements**:
   - Create a `requirements.txt` file with the following dependencies:
     ```
     langgraph
     langchain
     transformers
     fastapi
     gradio
     python-dotenv
     ```

4. **Deploy the Space**:
   - Push the code to your Space repository. Hugging Face will automatically install the dependencies and deploy the application.

5. **Access the Application**:
   - Once deployed, you can access the application via the provided URL (e.g., `https://huggingface.co/spaces/your-username/your-space-name`).

---

## Example Output

For the input text:
```
"The new iPhone 15 Pro was released yesterday by Apple in Cupertino. CEO Tim Cook presented the device."
```

The application might output:
- **Classification**: News
- **Entities**: Apple, Cupertino, Tim Cook
- **Summary**: Apple released the iPhone 15 Pro in Cupertino, presented by CEO Tim Cook.

---

## Limitations on Hugging Face Spaces
- **Text Length**: The free tier has a limit on input text length (2000 characters in this implementation).
- **Model Performance**: The `flan-t5-base` model is less powerful than `llama3.2`, so results may vary in quality.
- **Resource Constraints**: The application may run slower on the free tier due to limited computational resources.

---


### **Extra Points to Keep in Mind While Committing to Hugging Face**  
1. **Ensure SSH keys are loaded** before pushing:  
   ```sh
   ssh-add -l
   ```  
   If no keys are listed, add the key:  
   ```sh
   ssh-add ~/.ssh/id_ed25519
   ```  
2. **Start SSH Agent** if it's not running:  
   ```sh
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```  
3. **Verify SSH connection** with Hugging Face:  
   ```sh
   ssh -T git@hf.co
   ```  
   If it says **"Hi anonymous"**, SSH is not properly set up.  
4. **Use SSH instead of HTTPS for Git remote**:  
   ```sh
   git remote set-url origin git@hf.co:your-username/your-repo.git
   ```  
5. **Commit and push correctly**:  
   ```sh
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```  

---

### **Steps If `.bashrc` Is Not Updated**  
Run these commands manually each time before pushing:  
```sh
eval "$(ssh-agent -s)"
ssh-add -l
ssh-add ~/.ssh/id_ed25519
ssh -T git@hf.co
```

---

### **Code to Add in `.bashrc` for Persistent SSH Agent**  
Add the following to `~/.bashrc` to avoid running SSH agent commands every time:  
```sh
export PATH=$HOME/bin:$PATH

# Persistent SSH Agent Script
env=~/.ssh/agent.env

agent_load_env() {
    test -f "$env" && . "$env" >| /dev/null
}

agent_start() {
    (umask 077; ssh-agent >| "$env")
    . "$env" >| /dev/null
}

if [ -z "$SSH_AUTH_SOCK" ]; then
    agent_load_env
fi

if ! ps -ef | grep -q "[s]sh-agent"; then
    agent_start
fi

ssh-add -l > /dev/null 2>&1 || ssh-add ~/.ssh/id_ed25519 > /dev/null 2>&1
```

This ensures that the SSH agent starts automatically, and you don't need to manually run `ssh-add` every time.  

---


This modified version is optimized for Hugging Face Spaces and provides a lightweight, user-friendly text analysis tool. You can access the live version here:  
[Text Analysis Application on Hugging Face Spaces](https://huggingface.co/spaces/sankhadip10/textanalyzer)
