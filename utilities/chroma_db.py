import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import uuid
import os
from dotenv import load_dotenv
import json

load_dotenv()

def get_or_create_persistent_chromadb_client_and_collection(collection_name):

    chroma_client = chromadb.PersistentClient(f"./data/chromadb/{collection_name}")

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        ),
        metadata={"hnsw:space": "cosine"},
        get_or_create=True
    )

    return collection, chroma_client

def add_document_chunk_to_chroma_collection(collection_name, document_chunk, document_id=None):

    collection, chroma_client = get_or_create_persistent_chromadb_client_and_collection(collection_name)

    if document_id is None:
        guid = uuid.uuid4()
        guid_string = str(guid)
        document_id = guid_string

    collection.add(
        documents=[document_chunk],
        ids=[document_id]
    )

def query_chromadb_collection(collection_name, query, n_results):

    collection, chroma_client = get_or_create_persistent_chromadb_client_and_collection(collection_name)

    documents = collection.query(
        query_texts=[query],
        include=["documents", "metadatas"],
        n_results=n_results
    )

    if documents and documents['documents'] and documents['documents'][0]:

        return documents['documents'][0]

    else:

        return []

def delete_chromadb_collection(collection_name):
    
    collection, chroma_client = get_or_create_persistent_chromadb_client_and_collection(collection_name)
    
    try:
        chroma_client.delete_collection(name=collection_name)
        return (f"Collection '{collection_name}' has been successfully deleted.")

    except Exception as e:
        print(f"Error: {e}")
        return(f"An error occurred while trying to delete the collection: {e}")