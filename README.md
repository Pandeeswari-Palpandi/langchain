# LangChain learning examples

In this project, I learned the basics of building applications with LangChain.
I explored chat models, prompt templates, and used LangSmith to trace runs.
I learned how to create embeddings, load documents, and split text into chunks.
I practiced storing and retrieving relevant information with FAISS.

## Requirements

- Python installed and available in your terminal
- An OpenAI API key for the OpenAI model and embedding examples
- A LangSmith API key if you want to trace runs

## Setup

From the project directory, create and activate a virtual environment on
Windows:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the libraries used by the examples:

```powershell
python -m pip install langchain langchain-openai langchain-huggingface langchain-community langchain-text-splitters langsmith python-dotenv numpy faiss-cpu pypdf sentence-transformers
```

For the OpenAI and LangSmith examples, create a `.env` file in the project
directory with your own credentials:

```dotenv
OPENAI_API_KEY=your-openai-api-key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT_NAME=langchain-learning
```

Keep `.env` private; do not commit API keys to source control. OpenAI examples
require a valid API key and may incur usage charges. LangSmith tracing is
optional.

## Examples

Run an example from the project directory with `python <filename>`, for
example:

```powershell
python traced_app.py
```

| File | What it demonstrates |
| --- | --- |
| `chain.py` | A prompt, OpenAI chat model, and string output parser composed into a chain. |
| `langsmith_demo.py` | A single OpenAI chat-model chain that can be traced with LangSmith. |
| `traced_app.py` | Invoking one traced chat chain with several questions. |
| `openai-embeddings.py` | Creating a text embedding with OpenAI. |
| `huggingface-embeddings.py` | Creating an embedding with the `all-MiniLM-L6-v2` Hugging Face model. |
| `compare-meaning-with-vectors.py` | Comparing text embeddings using cosine similarity. |
| `document-loaders.py` | Loading text and PDF documents. The script creates `langchain.txt`; provide a valid `langchain.pdf` in this directory before running it. |
| `text-splitter.py` | Loading a text document and splitting it into overlapping chunks. |
| `document-split.py` | Splitting a string into overlapping character chunks. |
| `vector-store.py` | Creating a FAISS vector store, saving it locally, and searching it. |
| `load_from_local_vector_store.py` | Loading and querying the locally saved FAISS vector store. Run `vector-store.py` first. |
| `retriever.py` | Creating a FAISS-backed retriever and retrieving relevant text for a query. |

The Hugging Face embedding examples download their model the first time they
run. FAISS examples write/read a `vector_store` directory in the project
directory.

> `load_from_local_vector_store.py` enables FAISS dangerous deserialization
> because LangChain requires it to load this local index format. Only load a
> vector store that you created yourself or otherwise trust.
