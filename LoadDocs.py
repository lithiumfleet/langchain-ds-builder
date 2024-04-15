from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os



def load_docs_from_path(director:str="/data/lixubin/RetrieveQA/data/input", chunck_size=-1) -> list[Document]:
    """ load multiple files. """

    file_paths = list()
    if chunck_size == -1:
        chunck_size = 512
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunck_size)
    for item in os.walk(director):
        files = item[2]
        for i in files:
            file_paths.append(os.path.join(item[0], i))

    docs = list()
    for file in file_paths:
        assert os.path.isfile(file), f"{file} is not a loadable file."
        loader = TextLoader(file)
        doc = loader.load_and_split(text_splitter)
        assert isinstance(doc, list)
        docs += doc

    return docs



if __name__ == "__main__":
    docs = load_docs_from_path("./data/input")
    print()