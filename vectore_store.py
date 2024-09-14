
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter




def index_documents(urls, collection_name="rag-chroma", embedding_model="nomic-embed-text-v1.5"): # vector_db_dir
    """
    Function to load documents from URLs, split them, embed them, and store them in a Chroma vector database.

    Args:
    - urls (list): List of URLs to load documents from.
    - vector_db_dir : file path to save the vector store.
    - collection_name (str): Name of the vector database collection.
    - embedding_model (str): Name of the embedding model to use.

    Returns:
    - retriever: A retriever object that can be used to query the vector store.
    """
    
    # Load documents from URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]  # Flatten the list of documents

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to Chroma vector store with embeddings
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=NomicEmbeddings(model=embedding_model, inference_mode="local"),
        # persist_directory=vector_db_dir
    )

    # Return retriever for querying
    retriever = vectorstore.as_retriever()

    return retriever

