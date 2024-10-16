import streamlit as st
import os
import base64
import PyPDF2
import io

def upload_document(document_folder):
    uploaded_file = st.file_uploader("Choose a PDF file to upload", type=["pdf"])
    if uploaded_file is not None:
        if st.button("upload"):
            if not os.path.exists(document_folder):
                os.makedirs(document_folder)
            file_path = os.path.join(document_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"PDF file {uploaded_file.name} has been uploaded successfully!")

def read_document(document_folder, document_name):
    file_path = os.path.join(document_folder, document_name)
    if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            content = ""
            for page in range(num_pages):
                content += pdf_reader.pages[page].extract_text() + "\n\n"
        return content
    else:
        st.error(f"PDF file {document_name} not found in {document_folder}")

def chunk_document(document_folder, document_name, chunk_size=250, chunk_overlap=50):
    content = read_document(document_folder, document_name)
    if content:
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    else:
        st.error(f"Failed to read document {document_name}")
        return []

def download_document(document_folder, document_name):
    file_path = os.path.join(document_folder, document_name)
    if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        st.download_button(label=f"Download {document_name}", 
                           data=bytes_data, 
                           file_name=document_name, 
                           mime='application/pdf')
    else:
        st.error(f"PDF file {document_name} not found in {document_folder}")

def delete_document(document_folder, document_name):
    if st.button(f"Delete {document_name}"):
        file_path = os.path.join(document_folder, document_name)
        if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
            os.remove(file_path)
            st.success(f"PDF file {document_name} has been deleted successfully!")
        else:
            st.error(f"PDF file {document_name} not found in {document_folder}")