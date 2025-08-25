from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
from src.prompt import *
import os
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Cleaning function for model output
import re
def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r' +', ' ', text)
    # Remove /C followed by numbers
    text = re.sub(r'/C\d+', '', text)
    return text.strip()


app = Flask(__name__)

# Add a greeting document to the Pinecone index



load_dotenv()

Pinecone_API_KEY = os.getenv("Pinecone_API_KEY")
os.environ["PINECONE_API_KEY"] = Pinecone_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever= docsearch.as_retriever(search_type="similarity",search_kwargs={"k": 3})


generator = pipeline(    "text2text-generation", model="google/flan-t5-base")
chat_model = HuggingFacePipeline(pipeline=generator)
prompt= ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response = rag_chain.invoke({"input": input})
    cleaned_answer = clean_text(response["answer"])
    print("Response:", cleaned_answer)
    return str(cleaned_answer)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)