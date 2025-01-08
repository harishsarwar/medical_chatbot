import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

api_key = os.getenv("HUGGINGFACE_API_KEY")

os.environ["HUGGINGFACE_API_KEY"]=api_key 


data_path = "Medical_book.pdf"
DB_FAISS_PATH = "vectorstores/db_faiss"

embedding_model = "sentence-transformers/all-mpnet-base-v2" # Embeddig model from hugging face we can used random embedding model



# create vector database
def create_vecto_db():
    # pdf load.
    loader = PyPDFLoader(data_path)
    pages = loader.load()

    # splitting into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    # Huggingface embbeding model
    embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
    )

    # vector databases
    db = FAISS.from_documents(chunks,embeddings)
    db.save_local(DB_FAISS_PATH)



if __name__ == '__main__':
    create_vecto_db()