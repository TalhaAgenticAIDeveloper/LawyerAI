import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai



# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyDlGuiJOqQePVsQEu5gWiftb74RDGvcq-c")

# Streamlit Page Setup
st.set_page_config(page_title="LexAI", layout="wide")
st.title("📚 LexAI - AI-powered Legal Research Assistant")

# Sidebar for module selection
mode = st.sidebar.radio("📌 Select Module", ["Question Answering", "Explainer", "Analyzer"])

# PDF Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

# Text Chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)






# Embedding and Vector Store Creation
@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store





# Conversational Chain Setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the provided context." Do not provide a wrong answer.
    Context: {context}
    Question: {question}
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)







# QA Handler
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]




# File Upload (always visible)
pdf_docs = st.file_uploader("📄 Upload your PDF files", accept_multiple_files=True, type=["pdf"])





# PDF Processing
if st.button("🚀 Submit & Process PDF") and pdf_docs:
    with st.spinner("Processing PDF..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.session_state.raw_text = raw_text
        st.session_state.text_chunks = text_chunks
        st.success("✅ PDF processed successfully!")






# Warn if no document is processed
if "raw_text" not in st.session_state:
    st.info("📌 Please upload and process a PDF to continue.")
    st.stop()

raw_text = st.session_state.raw_text





# Module Logic
if mode == "Question Answering":
    st.subheader("🔎 Ask a Question")
    user_query = st.text_input("Ask a question about the uploaded document:")
    if user_query:
        with st.spinner("Processing your question..."):
            response = user_input(user_query)
            st.success("✅ Answer generated:")
            st.write(response)

elif mode == "Explainer":
    st.subheader("📘 Document Explainer")
    if st.button("Explain Document"):
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
        prompt = f"""
        First, determine if the following document is a legal document (such as agreements, contracts, policies, terms, or legal notices).

        If it **is NOT** a legal document, respond politely to the user:
        "This document does not appear to be a legal document. Please upload a legal document (e.g., a contract, agreement, or terms and conditions) for explanation."

        If it **IS** a legal document, then do the following:
        1. Provide a brief summary of the document.
        2. Extract and list all key legal clauses.
        3. Explain each clause in simple, plain English.
        4. Identify important entities involved (e.g., people, companies), dates, and obligations mentioned.

        Keep the explanation well-structured and easy to follow.

        Document content:
        {raw_text}
        """
        with st.spinner("Generating explanation..."):
            explanation = llm.invoke(prompt)
            st.success("✅ Explanation ready:")
            st.write(explanation.content)

elif mode == "Analyzer":
    st.subheader("🧠 Legal Risk Analyzer")
    if st.button("Analyze Document"):
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
        prompt = f"""
        Step 1: Determine whether the following document is a legal document (e.g., a contract, agreement, policy, terms and conditions, legal notice, etc.).

        - If it is NOT a legal document, respond respectfully with:
        "This document doesn't appear to be a legal document. Please upload a valid legal document (e.g., agreement, contract, terms, policy) for risk analysis."

        - If it IS a legal document, then proceed to analyze it and respond with clear and structured answers to the following:
        1. Are there any contradictory or conflicting clauses?
        2. Is any important section missing (e.g., signatures, termination, jurisdiction, payment terms)?
        3. Are there vague or ambiguous terms that could cause confusion or legal loopholes?
        4. Does the document include any elements that could be considered legally risky, non-compliant, or unfair?

        Use clear formatting and bullet points in your response for readability.

        Document:
        {raw_text}
        """
        with st.spinner("Analyzing document..."):
            analysis = llm.invoke(prompt)
            st.success("✅ Analysis complete:")
            st.write(analysis.content)
