import os
import glob
import re
import shutil

import argparse
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings

from config import MRKD_PATH


class VectorStore:
    def __init__(self, db_name:str, model_name:str="intfloat/multilingual-e5-large"):
        self.chroma_client = PersistentClient(path=db_name)
        self.collection = self.chroma_client.get_or_create_collection(name="urd", metadata={"hnsw:space": "cosine"})

        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def _extract_page_number(self, file_name) -> int:
        match = re.search(r'page_(\d+)\.md', file_name)

        if match:
            page_number = int(match.group(1))  # Extract the first group which is the number
            return page_number
        else:
            return None


    def upload_directory(self, dir_name:str):
        """Upload all markdown files contained in the directory"""

        dir_path = os.path.join(MRKD_PATH, dir_name)
        if not os.path.exists(dir_path):
            print(f"Directory not found in {MRKD_PATH}")
            return
        
        md_files = glob.glob(os.path.join(dir_path, '*.md'))
        print(f"Found {len(md_files)} file(s) in {dir_name}")
        print(f"Uploading file(s) to DB...")

        for filename in md_files:
            with open(filename, 'r', encoding='utf-8') as file:
                doc = file.read()
                self.collection.upsert(
                    documents=[doc],
                    embeddings=self.embedding_model.embed_documents([doc]),
                    ids=[os.path.basename(filename)],
                    metadatas=[{"dir_name":dir_name, "page":self._extract_page_number(filename), "path":filename}]
                )
        
        print("File(s) uploaded!")

    def query_db(self, question:str, dir_name:str, n_results:int=3):
        """Return the `n_results` closest documents relative to the question"""

        results = self.collection.query(
            query_embeddings=self.embedding_model.embed_documents([question]),
            n_results=n_results, # how many results to return
            where={"dir_name": dir_name} # Filter on the PDF to query
        )

        return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", help="Path to the directory", default="bce-bpce2023-urd-fr-mel-240415-11")

    args = parser.parse_args()
    dir_name = args.dir_name

    db_path = "test_db"

    vector_store = VectorStore(db_path)
    vector_store.upload_directory(dir_name)

    question = "Quel est le PNB de BPCE en 2023 ?"
    print()
    print("## QUESTION :")
    print(question)
    results = vector_store.query_db(question, dir_name, n_results=1)

    print("## RESULT : ")
    print('"""')
    print(results["documents"][0][0][0:100]+"...")
    print('"""')

    if os.path.exists(db_path):
        shutil.rmtree(db_path)