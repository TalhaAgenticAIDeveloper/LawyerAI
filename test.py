import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from crewai import Crew
import os
import google.generativeai as genai
from agents import LexAI_Agents
from tasks import LexAI_Tasks



# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyDlGuiJOqQePVsQEu5gWiftb74RDGvcq-c")

# Streamlit Page Setup
st.set_page_config(page_title="LexAI", layout="wide")
st.title("📚 LexAI - AI-powered Legal Research Assistant")

# Sidebar for module selection
mode = st.sidebar.radio("📌 Select Module", ["Question Answering", "Explainer", "Analyzer",  "Contract Drafting"])

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
        with st.spinner("Explaining your Document..."):

            agents = LexAI_Agents()
            Explainer = agents.Document_Explainer_Agent()

            tasks = LexAI_Tasks()
            Document_Explainer_Task = tasks.Document_Explanation_Task(Explainer, st.session_state.raw_text)

            crew = Crew(
                    agents=[Explainer],
                    tasks=[Document_Explainer_Task],
                )
            results = crew.kickoff()
            st.success("✅ Explanation Generated!")
            ai_response = results.raw
            st.write(ai_response)

elif mode == "Analyzer":
    st.subheader("🧠 Legal Risk Analyzer")
    if st.button("Analyze Document"):
        with st.spinner("Analyzing your Document..."):

            agents = LexAI_Agents()
            Analyzer = agents.Legal_Risk_Analyzer_Agent()

            tasks = LexAI_Tasks()
            Legal_Risk_Analyzer_Task = tasks.Legal_Risk_Analysis_Task(Analyzer, st.session_state.raw_text)


            crew = Crew(
                    agents=[Analyzer],
                    tasks=[Legal_Risk_Analyzer_Task],
                )
            results = crew.kickoff()
            st.success("✅ Analysis Generated!")
            ai_response = results.raw
            st.write(ai_response)


elif mode == "Contract Drafting":

    st.subheader("📝 Contract Drafting Assistant")

    # Step 1: Choose contract type
    contract_type = st.selectbox("Choose contract type:", ["Non-Disclosure Agreement (NDA)", "Employment Contract", "Service Agreement", "Lease Agreement", "Custom"])

    # Step 2: Optional fields to fill in
    party_a = st.text_input("Party A (e.g., Company Name)")
    party_b = st.text_input("Party B (e.g., Employee or Client Name)")
    jurisdiction = st.text_input("Jurisdiction (e.g., California, UK, Pakistan)")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    # Generate Contract
    if st.button("✍️ Generate Draft Contract"):
        with st.spinner("Generating contract draft..."):

            agents = LexAI_Agents()
            Drafter = agents.Contract_Drafting_Agent()

            tasks = LexAI_Tasks()
            # Contract_Drafting_Task = tasks.Contract_Drafting_Task(Drafter, st.session_state.text_chunks,)
            Contract_Drafting_Task = tasks.Contract_Drafting_Task(Drafter,contract_type, party_a, party_b, jurisdiction, start_date, end_date)

            crew = Crew(
                agents=[Drafter],
                tasks=[Contract_Drafting_Task],
            )
            results = crew.kickoff()
            st.spinner("Generating contract draft...")
            st.success("✅ Contract Draft Generated!")
            ai_response = results.raw
            st.write(ai_response)
