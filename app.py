import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load embedding model (still required for similarity)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load prebuilt FAISS vectors
vectors = FAISS.load_local(
    "vectors",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectors.as_retriever(search_kwargs={"k": 2})

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce",
    return_source_documents=True
)

st.title("PDF RAG Assistant")

query = st.text_area("Ask a question")

if query:
    with st.spinner("Thinking..."):
        result = chain.invoke({"query": query})
        st.write(result["result"])
