from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

text = """LangChain is an open-source framework used to build applications powered by Large Language Models (LLMs). 
    It enables LLMs to interact with external data sources, documents, databases, APIs, and tools.
    It supports features such as RAG, agents, prompt management, memory, and tool calling.
    LangChain is commonly used to develop AI chatbots, intelligent agents, document Q&A systems, and other GenAI applications."""

with open("langchain.txt", "w") as file:
    file.write(text)

print("Text written to langchain.txt")

textLoader = TextLoader("langchain.txt")
text_documents = textLoader.load()
print("Number of documents loaded:", len(text_documents))
print("Content of the first document:")
print(text_documents[0].page_content)

pdf_path = Path("langchain.pdf")
if not pdf_path.is_file():
    raise FileNotFoundError(f"Provide a valid PDF file at {pdf_path.resolve()}")

with pdf_path.open("rb") as pdf_file:
    if pdf_file.read(5) != b"%PDF-":
        raise ValueError(
            f"{pdf_path.resolve()} is not a PDF. Replace it with a valid PDF file."
        )

pdfLoader = PyPDFLoader(str(pdf_path))
pdf_documents = pdfLoader.load()
print("Number of PDF documents loaded:", len(pdf_documents))
print("Content of the first PDF document:")
print(pdf_documents[0].page_content)
