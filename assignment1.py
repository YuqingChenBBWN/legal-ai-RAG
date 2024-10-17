import streamlit as st
import os
from utilities.layout import page_config
from utilities.documents import upload_document, chunk_document, download_document, delete_document
from utilities.chroma_db import get_or_create_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection
from utilities.ai_embedding import text_small_embedding
from utilities.ai_inference import gpt4o_mini_inference

# Set page configuration
page_config()

# Constants
DOCUMENT_FOLDER = "documents"

# Ensure documents folder exists
if not os.path.exists(DOCUMENT_FOLDER):
    os.makedirs(DOCUMENT_FOLDER)

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = None

def process_uploaded_document(document_name):
    try:
        # Step 1: Chunk the document
        chunks = chunk_document(DOCUMENT_FOLDER, document_name)
        if not chunks:
            st.error(f"Failed to chunk document: {document_name}")
            return None
        
        # Step 2: Create or get collection
        collection_name = document_name.replace('.pdf', '')
        try:
            collection, _ = get_or_create_persistent_chromadb_client_and_collection(collection_name)
        except Exception as e:
            st.error(f"Failed to create/get collection: {str(e)}")
            return None
        
        # Step 3: Process each chunk
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding = text_small_embedding(chunk)
                
                # Add to collection
                add_document_chunk_to_chroma_collection(collection_name, chunk, f"chunk_{i}")
                
                # Update progress
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress)
                status_text.text(f"Processing progress: {int(progress * 100)}%")
            except Exception as e:
                st.error(f"Error processing chunk {i+1}: {str(e)}")
                continue  # Continue with the next chunk
        
        status_text.text("Document processing complete!")
        return collection
    except Exception as e:
        st.error(f"Unexpected error during document processing: {str(e)}")
        return None

def main():
    st.title("Legal Document Q&A System")

    # File upload
    upload_document(DOCUMENT_FOLDER)
    
    # Check if documents have been uploaded
    if os.path.exists(DOCUMENT_FOLDER):
        uploaded_files = [f for f in os.listdir(DOCUMENT_FOLDER) if f.lower().endswith('.pdf')]
        if uploaded_files:
            st.session_state.document_name = st.selectbox("Select a document to process", uploaded_files)
            
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    st.session_state.collection = process_uploaded_document(st.session_state.document_name)
                    if st.session_state.collection:
                        st.session_state.document_processed = True
                        st.success("Document processing complete!")
                    else:
                        st.error("Document processing failed. Please check the error messages above.")
        else:
            st.info("No PDF documents found. Please upload a PDF file first.")
    else:
        st.error(f"Folder '{DOCUMENT_FOLDER}' does not exist. Please ensure it has been created.")

    # User query input
    query = st.text_input("Enter your legal question:")

    if st.button("Submit Question"):
        if not st.session_state.document_processed:
            st.warning("Please upload and process a document first.")
        elif query:
            with st.spinner("Searching for an answer..."):
                collection_name = st.session_state.document_name.replace('.pdf', '')
                results = query_chromadb_collection(collection_name, query, 3)
                if results:
                    context = " ".join(results)
                    system_prompt = "You are a legal assistant. Answer the user's question based on the provided context."
                    instruction_prompt = f"Context: {context}\n\nQuestion: {query}\n\nPlease answer the question based on the context:"
                    answer = gpt4o_mini_inference(system_prompt, instruction_prompt)
                    st.subheader("Answer:")
                    st.write(answer)
                    st.subheader("Relevant document excerpts:")
                    for i, result in enumerate(results, 1):
                        st.write(f"{i}. {result[:200]}...")
                else:
                    st.info("No relevant answer found.")
        else:
            st.warning("Please enter a question.")

    # Document download and delete options
    if st.session_state.document_name:
        download_document(DOCUMENT_FOLDER, st.session_state.document_name)
        delete_document(DOCUMENT_FOLDER, st.session_state.document_name)

if __name__ == "__main__":
    main()