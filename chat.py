import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader

import time


PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Instruct: {instruct} 

Context:{document_context} 

Pages: {pages}

Output:
"""

PDF_STORAGE_PATH = 'data'
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="data")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
PIPE = pipeline("text-generation", model='data', tokenizer='data')
LANGUAGE_MODEL = HuggingFacePipeline(pipeline=PIPE)

def save_pdf(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf(file_path):
    document_loader = PyPDFLoader(file_path)
    return document_loader.load()

def create_chunks(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_processor.split_documents(raw_documents)

def add_chunks(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_similars(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query, k=3)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    context_pages = '  '.join(map(str, ["/".join(map(str, _)) for _ in [[int(doc.metadata.get(key)) for key in ['page', 'page_label']] for doc in relevant_docs]]))
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"instruct": user_query, "document_context": context_text, 'pages': context_pages})


st.title("PDF Chat")

uploaded_pdf = st.file_uploader(
    "Upload your PDF document",
    type="pdf",
    help="Select a PDF file for questions",
    accept_multiple_files=False)

if uploaded_pdf:
    saved_path = save_pdf(uploaded_pdf)
    raw_docs = load_pdf(saved_path)
    processed_chunks = create_chunks(raw_docs)
    add_chunks(processed_chunks)
    
    # st.success("Document uploaded successfully!")
    alert = st.success("Document uploaded successfully!") 
    time.sleep(5) 
    alert.empty()

    st.markdown("---")
    
    user_input = st.chat_input("Enter your question about the documents...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing..."):
            relevant_docs = find_similars(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant"):
            st.write(*ai_response.partition('Context:')[1:])


#----------------------------------------
#
# loader = PyPDFLoader(file_path)
# docs = loader.load()
#
# all_splits = text_splitter.split_documents(docs)
#
# retriever = faiss.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 2})
#
# from langchain.chains import RetrievalQA
#
# qa = RetrievalQA.from_chain_type(
#         llm = local_llm,
#         chain_type = "stuff",
#         retriever = retriever,
#         return_source_documents=True)
# generated_text = qa.invoke(query)
# answer = generated_text['result']