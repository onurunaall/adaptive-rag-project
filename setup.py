import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adaptive_rag_project",
    version="0.1.0",
    author="Onur Ünal",
    author_email="upklw@student.kit.edu",
    description="An implementation of Adaptive RAG with LLama3 and integrated workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/onurunaall/adaptive_rag_project",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain_community",
        "langchain-openai",
        "langchain_core",
        "langchain",
        "langgraph",
        "langchain-google-genai",
        "langchain-experimental",
        "langchain-nomic",
        "langchainhub",
        "langgraph.prebuilt",
        "streamlit",
        "tiktoken",
        "chromadb",
        "tavily-python",
        "BeautifulSoup4",
        "gpt4all",
        "pypdf",
        "PyPDF2",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0",
        "pydantic-settings>=2.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
