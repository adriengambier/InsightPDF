import base64
import os

import streamlit as st

from config import DB_PATH
import parse
import vector_store


PROMPT_TEMPLATE = """
    Réponds à la question en utilisant uniquement le contexte suivant :
    
    {context}
    
    --
    
    Réponds à la question en se basant sur le contexte ci-dessus: {question}
"""

@st.cache_resource
def init_vector_store(model_name="intfloat/multilingual-e5-large"):
    db = vector_store.VectorStore(DB_PATH, model_name=model_name)
    return db

@st.cache_resource
def init_parser():
    parser = parse.PDFParser()
    return parser

def display_pdf(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def dropdown_files_in_db(db):
    pdf_select = st.selectbox(
        "Select a file among the PDF already parsed",
        set([dir_name["dir_name"] for dir_name in db.collection.get()["metadatas"]]),
        index=None,
        placeholder="Choose a file...",
    )
    st.divider()

    return pdf_select