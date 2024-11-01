import os

import google.generativeai as genai
import pandas as pd
import streamlit as st

from config import DATA_PATH, MRKD_PATH
from config import GOOGLE_API_KEY
import utils

genai.configure(api_key=GOOGLE_API_KEY)


parser = utils.init_parser()
# "FinLang/finance-embeddings-investopedia"
# "intfloat/multilingual-e5-large"
db = utils.init_vector_store(model_name="FinLang/finance-embeddings-investopedia")
generative_model = genai.GenerativeModel("gemini-1.5-flash")

# Upload PDF
with st.sidebar:
    uploaded_file = st.file_uploader('Upload and parse a PDF file', type="pdf")
    
    if uploaded_file is not None:
        parse_btn = st.button("Parse!", type="primary")

        if parse_btn:
            # Parsing
            with st.spinner('Parsing PDF...'):
                is_parsed = parser.parse(uploaded_file, uploaded_file.name)

            if is_parsed:
                st.success("Document parsed!", icon="✅")
            else:
                st.warning("Document already parsed!")

            # Uploading
            filename, ext = os.path.splitext(uploaded_file.name)

            if len(db.collection.get(where={"dir_name":filename})["ids"]) == 0:
                with st.spinner('Uploading PDF to DB...'):
                    db.upload_directory(filename)

                st.success("Document uploaded to DB!", icon="✅")
            else:
                st.warning("Document already uploaded to DB!")           
            

st.title("📄 InsightPDF")

pdf_select = utils.dropdown_files_in_db(db)

if pdf_select is not None:

    question_tab, viewer_tab = st.tabs(["Question", "PDF Viewer"])
    
    with question_tab:
        with st.form(key='my_form'):
            question = st.text_input(label="Enter a question")
            n_results = st.number_input("Number of relevant documents to retrieve", value=1)
            submit_button = st.form_submit_button(label='Ask!')

        if submit_button:
            results = db.query_db(question, pdf_select, n_results=n_results)
            dist_list, metadatas_list = results["distances"][0], results["metadatas"][0]
            res_df = pd.DataFrame.from_dict({
                "Markdown directory":[metadata["dir_name"] for metadata in metadatas_list],
                "Page":[metadata["page"] for metadata in metadatas_list],
                "Distance":dist_list
            })
            res_df.index += 1

            query = utils.PROMPT_TEMPLATE.format(context="\n\n---\n\n".join([doc for doc in results["documents"][0]]), question=question)
            response = generative_model.generate_content(query)

            st.subheader("Answer")
            st.markdown(response.text)

            st.divider()
            st.subheader("Sources")
            st.dataframe(res_df)

    with viewer_tab:
        pdf_path = os.path.join(DATA_PATH, pdf_select + ".pdf")
        utils.display_pdf(pdf_path)