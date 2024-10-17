import streamlit as st
import os

def get_credentials():
    with st.sidebar:
        st.markdown("## API Credentials")
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return True
        else:
            st.warning("Please enter your OpenAI API key")
            return False