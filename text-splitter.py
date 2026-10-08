from langchain_text_splitters import RecursiveCharacterTextSplitter

long_text = """Python is a popular, high-level, general-purpose programming language known for its simplicity and readability. 
It was created by Guido van Rossum and first released in 1991. 
Python uses a clean and easy-to-understand syntax, making it suitable for beginners and experienced developers. 
It supports multiple programming paradigms, including object-oriented, procedural, and functional programming. 
Python is widely used for web development, data analysis, artificial intelligence, machine learning, automation, and scripting. 
Popular frameworks such as Django, Flask, and FastAPI are used to build web applications and APIs. 
Python provides a large collection of libraries and frameworks that speed up software development. 
It is also commonly used in AI and GenAI applications because of libraries such as NumPy, Pandas, PyTorch, and LangChain. 
Python is cross-platform and can run on Windows, Linux, and macOS. Due to its flexibility, large community, 
and extensive ecosystem, Python is one of the most widely used programming languages today."""

print("Original text length:", len(long_text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(long_text)
print("Number of chunks:", len(chunks))
print("Content of the first chunk:")
print(chunks[0])
