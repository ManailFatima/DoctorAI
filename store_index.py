from langchain_pinecone import PineconeVectorStore
from datasets import load_dataset
from langchain.schema import Document
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
import os


Pinecone_API_KEY = os.getenv("Pinecone_API_KEY")
os.environ["PINECONE_API_KEY"] = Pinecone_API_KEY


extracted_data= load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_splitter(minimal_docs)


embeddings = download_embeddings()

pinecone_API_key=Pinecone_API_KEY
pc = Pinecone(api_key=pinecone_API_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))
index = pc.Index(index_name) 



dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
docs = []
for item in dataset["train"]:
    question = item.get("instruction", "") + " " + item.get("input", "")
    answer = item.get("output", "")
    docs.append(Document(
        page_content=f"Q: {question}\nA: {answer}",
        metadata={"source": "huggingface_lavita/ChatDoctor-HealthCareMagic-100k"}
    ))



docssearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


def chunked_upload(docs, batch_size=50):
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        docsearch.add_documents(batch)
chunked_upload(docs, batch_size=50)


docsearch.add_documents(docs)
