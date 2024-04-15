from LoadDocs import load_docs_from_path
from MakeQA import make_QA_pair, to_json_file

docs = load_docs_from_path()
make_QA_pair(docs, to_json_file)
