# filename: app.py

import os
import glob
from dotenv import load_dotenv
import gradio as gr
import openai
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings

# Constants
MODEL = "llama-3.3-70b-versatile"
db_name = "vector_db"

# Load environment variables
load_dotenv()
openai.api_key = os.environ.get("GROQ_API")
openai.base_url = "https://api.groq.com/openai/v1"

# Document loading
folders = glob.glob("knowledge-base/*")

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

# Text splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
vectorstore.persist()

# Vector info
collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

# Metadata and color assignment
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
metadatas = result['metadatas']
doc_types = [metadata['doc_type'] for metadata in metadatas]

color_map = {'products': 'blue', 'employees': 'green', 'contracts': 'red', 'company': 'orange'}
colors = [color_map.get(t, 'gray') for t in doc_types]

# LLM setup
llm = ChatOpenAI(
    temperature=0.7,
    model_name=MODEL,
    openai_api_key=os.environ.get("GROQ_API"),
    openai_api_base="https://api.groq.com/openai/v1"
)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Chat function
def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

# Launch Gradio chat interface
if __name__ == "__main__":
    gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
