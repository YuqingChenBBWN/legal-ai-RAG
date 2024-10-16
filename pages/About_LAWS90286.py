import streamlit as st
from utilities.ai_embedding import text_small_embedding
from utilities.ai_inference import gpt4o_mini_inference, gpt4o_mini_inference_yes_no
from utilities.chroma_db import get_or_create_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection, delete_chromadb_collection
from utilities.documents import upload_document, read_document, chunk_document, download_document, delete_document
from utilities.layout import page_config

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = None

if "instruction_prompt" not in st.session_state:
    st.session_state.instruction_prompt = None

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

if "chunk" not in st.session_state:
    st.session_state.chunk = None

if "document_folder" not in st.session_state:
    st.session_state.document_folder = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

page_config()

st.markdown("## About LAWS90286")

st.write("")

st.markdown("### AI Utilities")

st.markdown("""Under the `utilities` folder you have the following capabilties:
- ai_inference.py --> how to call the OpenAI to generate a response
- ai_embedding.py --> how to call the OpenAI API to create an embedding
""")

with st.expander("ai_inference - normal", expanded=False):

    st.markdown("""ai_inference.py contains a function:
    `gpt4o_mini_inference(system_prompt, instruction_prompt)`
    You use this function by replacing system_prompt and instruction_prompt with text telling the AI what to do.
    You can try it out below:
    """)

    st.session_state.system_prompt = st.text_input("System Prompt", key="ai_system_normal")

    st.session_state.instruction_prompt = st.text_input("Instruction Prompt", key="ai_instruction_normal")

    if st.session_state.system_prompt is not None and st.session_state.instruction_prompt is not None:

        st.markdown(f"""The code looks like:
        `response = gpt4o_mini_inference({st.session_state.system_prompt}, {st.session_state.instruction_prompt})`
        """)

        if st.button("response"):

            response = gpt4o_mini_inference(st.session_state.system_prompt, st.session_state.instruction_prompt)

            st.markdown(response)

with st.expander("ai_inference - classification", expanded=False):

    st.markdown("""ai_inference.py contains a function:
    `gpt4o_mini_inference_yes_no(system_prompt, instruction_prompt)`
    You use this function by replacing system_prompt and instruction_prompt with text telling the AI what to do.
    This function is unique because it will only ever respond 'yes' or 'no'.
    You can try it out below:
    """)

    st.session_state.system_prompt = st.text_input("System Prompt", key="ai_system_classify")

    st.session_state.instruction_prompt = st.text_input("Instruction Prompt", key="ai_instruction_classify")

    if st.session_state.system_prompt is not None and st.session_state.instruction_prompt is not None:

        st.markdown(f"""The code looks like:
        `response = gpt4o_mini_inference_yes_no({st.session_state.system_prompt}, {st.session_state.instruction_prompt})`
        """)

        if st.button("classify"):

            response = gpt4o_mini_inference_yes_no(st.session_state.system_prompt, st.session_state.instruction_prompt)

            st.markdown(response)

st.markdown("### Chroma Database Utilities")

st.markdown("""Under the `utilities` folder you have the following capabilties:
- chroma_db.py --> how to create and delete a vector database, and how to use a vector database to store and retrieve text
""")

with st.expander("add chunk to collection", expanded=False):

    st.markdown("""chroma_db.py contains a function:
    `add_document_chunk_to_chroma_collection(collection, document_chunk, document_id=None)`
    You use this function to add new chunks to a chroma database collection.
    You can try it out below:
    """)

    st.session_state.collection_name = st.text_input("Collection", key="collection_chunk")

    st.session_state.chunk = st.text_input("Chunk", key="chunk_to_add")

    if st.session_state.collection_name is not None and st.session_state.chunk is not None:

        st.markdown(f"""The code looks like:
        `add_document_chunk_to_chroma_collection({st.session_state.collection_name}, {st.session_state.chunk})`
        """)

        if st.button("add chunk to collection"):

            add_document_chunk_to_chroma_collection(st.session_state.collection_name, st.session_state.chunk)

with st.expander("query collection", expanded=False):

    st.markdown("""chroma_db.py contains a function:
    `query_chromadb_collection(collection, query, n_results)`
    You use this function to retrieve chunks from a chroma database collection.
    You can try it out below:
    """)

    st.session_state.collection_name = st.text_input("Collection", key="collection_search")

    st.session_state.search = st.text_input("Query", key="search_query")

    st.session_state.num_queries = st.number_input("Number of Results", key="collection_results", step=1)

    if st.session_state.collection_name is not None and st.session_state.search is not None and st.session_state.num_queries is not None:

        st.markdown(f"""The code looks like:
        `query_chromadb_collection({st.session_state.collection_name}, {st.session_state.search}, {st.session_state.num_queries})`
        """)

        if st.button("query collection"):

            results = query_chromadb_collection(st.session_state.collection_name, st.session_state.search, st.session_state.num_queries)

            st.write(results)

with st.expander("delete collection", expanded=False):

    st.markdown("""chroma_db.py contains a function:
    `delete_chromadb_collection(collection_name)`
    You use this function to delete a chroma database collection.
    You can try it out below:
    """)

    st.session_state.collection_name = st.text_input("Name", key="collection_delete")

    if st.session_state.collection_name is not None:

        st.markdown(f"""The code looks like:
        `delete_chromadb_collection({st.session_state.collection_name})`
        """)

        if st.button("delete collection"):

            outcome = delete_chromadb_collection(st.session_state.collection_name)

            st.markdown(outcome)

st.markdown("### Documents Utilities")

st.markdown("""Under the `utilities` folder you have the following capabilties:
- documents.py --> how to upload, read, chunk, download, and delete documents
""")

with st.expander("upload_document", expanded=False):

    st.markdown("""`upload_document(document_folder)`
    You use this function by replacing document_folder with text telling the function what folder to save the document in.
    You can try it out below:
    """)

    st.session_state.document_folder = st.text_input("Document Folder", key="folder_upload")

    if st.session_state.document_folder is not None:

        st.markdown(f"""The code looks like:
        `upload_document({st.session_state.document_folder})`
        """)

        upload_document(st.session_state.document_folder)

with st.expander("read_document", expanded=False):

    st.markdown("""`read_document(document_folder, document_name)`
    You use this function by replacing document_folder and document_name with text telling the function where to find the document.
    You can try it out below:
    """)

    st.session_state.document_folder = st.text_input("Document Folder", key="folder_read")

    st.session_state.document_name = st.text_input("Document Name", key="name_read")

    if st.session_state.document_folder is not None and st.session_state.document_name is not None:

        st.markdown(f"""The code looks like:
        `document = read_document({st.session_state.document_folder}, {st.session_state.document_name})`
        """)

        if st.button("read"):

            document = read_document(st.session_state.document_folder, st.session_state.document_name)

            st.markdown(document)
        
with st.expander("chunk_document", expanded=False):

    st.markdown("""`chunk_document(document_folder, document_name, chunk_size=250, chunk_overlap=50)`
    You use this function by replacing document_folder and document_name with text telling the function where to find the document.
    The function returns a list of the chunks. You can iterate over these using a `for-loop`.
    You can try it out below:
    """)

    st.session_state.document_folder = st.text_input("Document Folder", key="folder_chunk")

    st.session_state.document_name = st.text_input("Document Name", key="name_chunk")

    if st.session_state.document_folder is not None and st.session_state.document_name is not None:

        st.markdown(f"""The code looks like:
        `list_of_chunks = chunk_document({st.session_state.document_folder}, {st.session_state.document_name}, 250, 50)`
        """)

        if st.button("chunk"):

            list_of_chunks = chunk_document(st.session_state.document_folder, st.session_state.document_name)

            st.markdown(list_of_chunks)

with st.expander("download_document", expanded=False):

    st.markdown("""`download_document(document_folder, document_name)`
    You use this function by replacing document_folder and document_name with text telling the function where to find the document.
    You can try it out below:
    """)

    st.session_state.document_folder = st.text_input("Document Folder", key="folder_download")

    st.session_state.document_name = st.text_input("Document Name", key="name_download")

    if st.session_state.document_folder is not None and st.session_state.document_name is not None:

        st.markdown(f"""The code looks like:
        `document = download_document({st.session_state.document_folder}, {st.session_state.document_name})`
        """)

        download_document(st.session_state.document_folder, st.session_state.document_name)

with st.expander("delete_document", expanded=False):

    st.markdown("""`delete_document(document_folder, document_name)`
    You use this function by replacing document_folder and document_name with text telling the function where to find the document.
    You can try it out below:
    """)

    st.session_state.document_folder = st.text_input("Document Folder", key="folder_delete")

    st.session_state.document_name = st.text_input("Document Name", key="name_delete")

    if st.session_state.document_folder is not None and st.session_state.document_name is not None:

        st.markdown(f"""The code looks like:
        `delete_document({st.session_state.document_folder}, {st.session_state.document_name})`
        """)

        delete_document(st.session_state.document_folder, st.session_state.document_name)

st.write("")
st.markdown("### General Coding Requirements")
st.write("")

st.write("")
st.markdown("##### Creating Variables")
st.write("")

st.markdown("""
Variables hold information temporarily in your code. Ordinarily in streamlit we use `st.session_state`.
Create a st.session_state variable as follows:
```
if "<insert name>" not in st.session_state:
    st.session_state.<insert name> = None
```

Then set your st.session_state variable as:
```
st.session_state.<insert name> = ...
```
""")

st.write("")
st.markdown("##### Calling Functions")
st.write("")

st.markdown("""
Functions are called by stating the function name and including the inputs required of the function.
If a function returns a value (i.e., has an output as well as an input), that output can be saved to a variable.
For example:
```
st.session_state.my_list_of_chunks = chunk_document(document_folder, document_name)
```
""")

st.write("")
st.markdown("##### Testing Conditions")
st.write("")

st.markdown("""
Conditionals are tested with 'if / else' statements.
One way to do this is in combination with the `gpt4o_mini_inference_yes_no(system_prompt, instruction_prompt)` function.
For example:
```
is_contract = gpt4o_mini_inference_yes_no("You are a lawyer.", "Your task is to determine whether the following document is a contract: ...")

if is_contract == "yes":

    <do some thing here>

else:

    <do some other thing here>
```
""")

st.write("")
st.markdown("##### Iterating")
st.write("")

st.markdown("""
Iterating is a way of doing multiple things automatically to items that are in a datastructure that can be iterated over (like a list).
For example, if you had a list of chunks you could do the same thing using each chunk:

```
list_of_chunks = ['A', 'B', 'C', 'D']

for chunk in list_of_chunks:

    add_document_chunk_to_chroma_collection(chunk)
```

The code above would add each chunk individual to the Chroma database.
""")