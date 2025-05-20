import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
tracing_enabled = os.getenv('LANGCHAIN_TRACING_V2')
endpoint = os.getenv('LANGCHAIN_ENDPOINT')
api_key = os.getenv('LANGCHAIN_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

import uuid
import os
from urllib.parse import urlparse

from langchain_chroma import Chroma

def create_vectorstore(documents, embeddings, source):
    """
    Creates a vector store with a collection name based on the source.

    Args:
        documents (list): List of document splits.
        embeddings: The embeddings model.
        source (str): The source of the document (file path or URL).

    Returns:
        vectorstore: The created vector store.
    """
    
    # Determine collection name based on the source type
    if source.startswith("http"):
        # If it's a URL, extract the domain name
        parsed_url = urlparse(source)
        collection_name = parsed_url.netloc.replace('.', '_')
    else:
        # If it's a file, use the filename without extension
        filename = os.path.basename(source)
        collection_name = os.path.splitext(filename)[0]
    
    # Append a short UUID for uniqueness
    collection_name = f"{collection_name}_{uuid.uuid4().hex[:4]}"
    
    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        collection_name=collection_name
    )
    
    print(f"[INFO] Vectorstore created with collection name: {collection_name}")
    return vectorstore

# === Import Libraries ===
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# === Initialize Gemini LLM ===
llm = GoogleGenerativeAI(
    model="models/gemini-1.5-pro-002",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

def embed_data(url):
    """
    Function to load, chunk, and embed data from a given URL.
    
    Args:
        url (str): The web page URL to scrape and embed.
        
    Returns:
        List[dict]: A list of embeddings with their corresponding text chunks.
    """
    print(f"[INFO] Loading documents from: {url}")
    
    # Load Documents
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.BeautifulSoup.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} documents from the URL.")

    # Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"[INFO] Split into {len(splits)} chunks.")

    # Embedding with Gemini
    print("[INFO] Generating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    create_vectorstore(splits, embeddings, url)

import json
from langchain_core.documents import Document


def embed_data_from_json(file_path, content_key="content"):
    """
    Function to load and embed data from a JSON file without splitting.

    Args:
        file_path (str): The path to the JSON file.
        content_key (str): The key to extract content from. Default is "content".

    Returns:
        vectorstore: The created vector store.
    """
    print(f"[INFO] Loading documents from JSON: {file_path}")

    # Load JSON Data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract Content and wrap as Document objects
    documents = [Document(page_content=item[content_key]) for item in data if content_key in item]
    print(f"[INFO] Loaded {len(documents)} documents from the JSON file.")

    # Embedding with Gemini
    print("[INFO] Generating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vectorstore1 = create_vectorstore(documents, embeddings, file_path)
    return vectorstore1

import os
from openai import vector_stores
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb import Client
from chromadb.config import Settings
from langchain_community.document_loaders import PDFPlumberLoader
def embed_data_from_pdf(file_path):
    """
    Function to load, chunk, embed, and store data from a PDF file.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        retriever: A ChromaDB retriever for semantic search.
    """
    print(f"[INFO] Loading PDF document from: {file_path}")
    
    # Load PDF Data
    loader = PDFPlumberLoader("de-an-tuyen-sinh-2024final.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(documents)

    # Initialize Gemini Embeddings
    print("[INFO] Generating embeddings with Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # Store embeddings in Chroma Vector Store
    print("[INFO] Storing embeddings in Chroma Vector Store...")
    print(splits[0])
    print(splits[1])
    print(splits[2])
    vectorstore1 = create_vectorstore(splits, embeddings, file_path)

    # Create Retriever
    print("[INFO] Creating retriever...")
    retriever = vectorstore1.as_retriever()

    # Print Summary
    print("[INFO] PDF embedding and storage complete.")
    print(f"[INFO] Total Documents Indexed: {len(splits)}")
    print(f"[INFO] Index Name: vietnam_university_pdfs")
    print(f"[INFO] Source: {file_path}")
    
    return retriever

def auto_embed(source):
    """
    Automatically calls the appropriate embed function based on the source type.

    Args:
        source (str): URL or file path (JSON or PDF).

    Returns:
        The result of the corresponding embed function.
    """
    if source.startswith("http://") or source.startswith("https://"):
        print("[INFO] Detected URL. Using embed_data for web content.")
        return embed_data(source)
    elif source.lower().endswith(".json"):
        print("[INFO] Detected JSON file. Using embed_data_from_json.")
        return embed_data_from_json(source)
    elif source.lower().endswith(".pdf"):
        print("[INFO] Detected PDF file. Using embed_data_from_pdf.")
        return embed_data_from_pdf(source)
    else:
        print(f"[WARNING] Unsupported source type: {source}. Skipping.")
        return None