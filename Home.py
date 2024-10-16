__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from utilities.ai_embedding import text_small_embedding
from utilities.ai_inference import gpt4o_mini_inference, gpt4o_mini_inference_yes_no
from utilities.chroma_db import get_or_create_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection, delete_chromadb_collection
from utilities.documents import upload_document, read_document, chunk_document, download_document, delete_document
from utilities.layout import page_config

page_config()

st.markdown("## LAWS90286 Template")
