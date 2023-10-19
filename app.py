#app.py
from flask import Flask, flash, request, redirect, url_for, render_template

from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import os

import pickle
os.environ["OPENAI_API_KEY"] = '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'

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024* 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','pdf'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def read_pdf(pdf):
    print(pdf,"pdf")
    text = ""
    for page in PdfReader(pdf).pages:
        text+= page.extract_text()
    # print(text)
    # return text, pdf.name   
    return text
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
# pdf_text=read_pdf("/home/jesliya-j/chatbot_summ/static/uploads/1.pdf")
# knowledgeBase = process_text(pdf_text)
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        text="kkkkk"
        print("req came")
        try:
            text = request.form['text']
            print(text)
            # query="give me summary"
            query=text
        except:
            pass
            # return"success"
        # pdf_text=read_pdf("/home/jesliya-j/chatbot_summ/static/uploads/1.pdf")
        # knowledgeBase = process_text(pdf_text)
        try:
            with open('config_dictionary', 'rb') as config_dictionary_file:
                knowledgeBase = pickle.load(config_dictionary_file)
                print(knowledgeBase,"type..............")
                docs = knowledgeBase.similarity_search(query)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type='stuff')
        
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                flash("QUESTION  : "+query)
                flash("ANSWER    : "+response)
                filename="static/uploads/aa.jpeg"
            # filenameos.path.join(app.config['UPLOAD_FOLDER'], filename
                return render_template('index2.html', filename=filename)
            # return redirect(request.url)
        except:
            flash("PLease upload the pdf first")
            return render_template('index.html', filename=filename)
    file = request.files['file']
    if file.filename == '':
        flash('No pdf selected for uploading')
        # return redirect(request.url)
        filename=''
        return render_template('index.html', filename=filename)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename,"filename")
        filename="aa.jpeg"
        print(filename,"filename")
        filename2="1.pdf"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        pdf_text=read_pdf(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        knowledgeBase = process_text(pdf_text)
        with open('config_dictionary', 'wb') as config_dictionary_file:
 
            pickle.dump(knowledgeBase, config_dictionary_file)
        #print('upload_image filename: ' + filename)
        text="Knowledge base created. ask any question"
        flash(text)
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    # return "hello"
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(host="0.0.0.0")