import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def load_documents():
    loader = PyPDFLoader("indian_govt_schemes.pdf")  
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    return docs


@st.cache_resource
def create_vector_db():
    docs = load_documents()
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb


@st.cache_resource
def build_qa_chain():
    vectordb = create_vector_db()
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "language"],
        template="""
        You are a helpful assistant who explains Indian government schemes in simple {language}.
        Use the following context to answer the question accurately:

        Context:
        {context}

        Question:
        {question}

        Answer in {language}:
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )
    return qa_chain


st.set_page_config(page_title="Multilingual Govt Scheme Chatbot 🇮🇳", layout="centered")
st.title(" 🗣 Ask about Indian Government Schemes in Tamil / Hindi / English")

language = st.selectbox("Choose Language:", ["English", "Tamil", "Hindi"])
user_query = st.text_input("Ask a question about any Indian Govt Scheme:")

if user_query:
    with st.spinner("Fetching answer from govt scheme documents..."):
        qa = build_qa_chain()
        final_input = {
            "question": user_query,
            "context": "",
            "language": language
        }
        response = qa.run(final_input)
        st.markdown("### 📘 Answer:")
        st.success(response)

st.markdown("---")
st.markdown("⚡ Powered by OpenAI + LangChain + Streamlit + FAISS")