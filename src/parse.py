import os

import argparse
from llama_parse import LlamaParse
from typing import Union
from io import BufferedIOBase

from config import LLAMA_PARSE_API_KEY, DATA_PATH, MRKD_PATH


class PDFParser:
    def __init__(self):
        self.parser = LlamaParse(
            api_key=LLAMA_PARSE_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
            result_type="markdown",  # "markdown" and "text" are available
            verbose=True,
            language="fr",  # Optionally you can define a language, default=en
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt-4o-mini"
        )

    def parse(self, pdf_obj:Union[str, bytes, BufferedIOBase], pdf_name:str) -> int:
        name = os.path.splitext(pdf_name)[0] # Remove ".pdf" from pdf_name
        output_path = os.path.join(MRKD_PATH, name)

        if name in os.listdir(MRKD_PATH):
            print(f"{name} already parsed!")
            return 0
        
        if type(pdf_obj) != str:
            with open(os.path.join(DATA_PATH, pdf_name), 'wb') as f: 
                f.write(pdf_obj.getvalue())
        
        documents_parsed = self.parser.load_data(pdf_obj, extra_info={"file_name":pdf_name})
    
        os.makedirs(output_path)
        
        for page in range(len(documents_parsed)):
            print(f"Writing page {page+1}/{len(documents_parsed)}")
            with open(os.path.join(output_path, f"{name}_page_{page+1}.md"), 'w', encoding='utf-8') as file:
                file.write(documents_parsed[page].text)

        print(f"PDF parsed and saved to {output_path}")

        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the PDF", default=os.path.join(DATA_PATH, "bce-bpce2023-urd-fr-mel-240415-11.pdf"))
    
    args = parser.parse_args()
    path = args.path

    pdf_parser = PDFParser()
    pdf_parser.parse(path, os.path.basename(path))
