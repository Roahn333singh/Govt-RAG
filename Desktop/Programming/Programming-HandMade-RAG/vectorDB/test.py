import os
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from fastembed import SparseTextEmbedding


def load_documents(file_path):
    loader=PyPDFLoader(file_path)
    return loader.load()

def split_document(docs,chunk_size=500,chunk_overlap=50):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,  
    )

    return splitter.split_documents(docs)

if __name__=="__main__":
    # print(split_document(load_documents("../data/FMISC.pdf")))
    files = [f for f in os.listdir("../data") if os.path.isfile(os.path.join("../data", f))]
    print(files)