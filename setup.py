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
    # Pinning versions to ensure compatibility and consistent CI runs
    install_requires=[
        "beautifulsoup4~=4.12.3",
        "chromadb~=0.5.0",
        "gpt4all~=2.7.0",
        "langchain~=0.2.5",
        "langchain-community~=0.2.5",
        "langchain-core~=0.2.9",
        "langchain-experimental~=0.0.61",
        "langchain-google-genai~=1.0.6",
        "langchain-openai~=0.1.7",
        "langchainhub~=0.1.20",
        "langgraph~=0.1.0",
        "pydantic~=2.7.4",
        "pydantic-settings~=2.3.4",
        "pypdf~=4.2.0",
        "python-dotenv~=1.0.1",
        "streamlit~=1.35.0",
        "tavily-python~=0.3.3",
        "tiktoken~=0.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.9",
)
