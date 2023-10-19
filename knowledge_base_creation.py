from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import os

import pickle
os.environ["OPENAI_API_KEY"] = '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'

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    # embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    # vectorstore = FAISS.from_documents(all_splits, embeddings)
    
    return knowledgeBase

file1 = open("text_extracted.txt","r+")
pdf_text=file1.read()
file1.close()
knowledgeBase = process_text(pdf_text)
with open('config_dictionary', 'wb') as config_dictionary_file:
    pickle.dump(knowledgeBase, config_dictionary_file)