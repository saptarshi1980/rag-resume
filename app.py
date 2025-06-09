import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Resume RAG Assistant")
st.title("📄 Resume QA with RAG")

with st.form("resume_form"):
    uploaded_file = st.file_uploader("Upload a resume PDF", type="pdf")
    query = st.text_input("Ask a question about the resume:")
    submitted = st.form_submit_button("Submit")

if submitted and uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()

    embedding = HuggingFaceEmbeddings(model_name="./local_embedding_model")
    vectorstore = FAISS.from_documents(documents, embedding)

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.success("Answer:")
        st.write(result["result"])
