import streamlit as st
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import google.generativeai as genai
import base64
import editdistance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tempfile import NamedTemporaryFile

# Constants for file paths and directories
DEFAULT_PDF_PATH = Path("data/The_Gift_of_the_Magi.pdf")
DEFAULT_OUTPUT_PATH = Path("output/extracted_text.txt")
DEFAULT_SUMMARY_PATH = Path("output/summary.txt")
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
PERSIST_DIRECTORY = "chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# --- pdf_to_text_extraction.py code ---

def setup_gemini_api(api_key):
    genai.configure(api_key=api_key)

def pdf_to_base64(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
        return pdf_data
    except FileNotFoundError:
        st.error(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred during PDF to base64 conversion: {e}")
        return None

def extract_text_from_pdf(pdf_path, api_key):
    setup_gemini_api(api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    pdf_base64 = pdf_to_base64(pdf_path)
    if pdf_base64 is None:
        return None

    pdf_data = {
        "mime_type": "application/pdf",
        "data": pdf_base64,
    }

    try:
        response = model.generate_content([
            "Extract all text from the given PDF. Return the plain text.",
            pdf_data
        ])
        return response.text
    except Exception as e:
        st.error(f"An error occurred during OCR: {e}")
        return None

def calculate_cer(ground_truth_file, ocr_text_file):
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as gt_file:
            ground_truth = gt_file.read().replace("\n", "").replace(" ", "")
        with open(ocr_text_file, 'r', encoding='utf-8') as ocr_file:
            ocr_text = ocr_file.read().replace("\n", "").replace(" ", "")
        edit_distance = editdistance.eval(ground_truth, ocr_text)
        n = len(ground_truth)
        cer = edit_distance / n if n != 0 else 0.0
        return cer
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during CER calculation: {e}")
        return None
    
# Functions for PDF extraction and summarization
def process_pdf(pdf_path, output_path, summary_path, api_key):
    """Processes a PDF, extracts text, summarizes, and returns paths."""
    setup_gemini_api(api_key)  # Setup Gemini API with the given key
    extracted_text = extract_text_from_pdf(pdf_path, api_key)
    if not extracted_text:
        st.error("Error extracting text from PDF.")
        return None, None

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    summary = summarize_text(extracted_text, api_key)
    if not summary:
        st.error("Error summarizing text.")
        return None, None

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return output_path, summary_path


# --- summarization.py code ---

def summarize_text(text, api_key):
    """Summarizes the given text using Gemini."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        response = model.generate_content(
            f"Provide a summary of the following document, including: 1. The main topic or theme. 2. The key supporting points or arguments. 3. The overall conclusion or message: {text}"
        )
        return response.text
    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")
        return None

# --- rag.py code ---

def load_and_chunk_text(file_path):
    """Load text from file and chunk it into smaller pieces."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        return texts
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        return []

def create_vectorstore(file_path):
    """Creates or loads a Chroma vectorstore from text chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if "vectorstore" in st.session_state and os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            return vectorstore

        texts = load_and_chunk_text(file_path)
        if not texts:
            st.error("No text chunks created from the document.")
            return None
        vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
        st.session_state["vectorstore"] = vectorstore
        return vectorstore
    except Exception as e:
        st.error(f"Error creating/loading vectorstore: {e}")
        return None

def retrieve_relevant_chunks(vectorstore, query):
    """Retrieve relevant chunks from the vectorstore based on the query."""
    if not vectorstore:
        return []
    try:
        relevant_chunks = vectorstore.similarity_search(query)
        return relevant_chunks
    except Exception as e:
        st.error(f"Error in retrieving relevant chunks: {e}")
        return []

def generate_prompt(relevant_chunks, query):
    """Generate a prompt using the relevant chunks of text and the user's query."""
    if not relevant_chunks:
        return "No relevant information found in the document."
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    prompt = f"""
    You are a helpful chatbot that answers questions based on the provided document context.
    Context: {context}
    Question: {query}
    Answer:
    """
    return prompt

def setup_model(api_key):
    """Configure API key for Gemini and initialize the model."""
    if not api_key:
        raise ValueError("Gemini API key is missing.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def generate_response(model, prompt):
    """Generate a response using the Gemini model."""
    try:
        if isinstance(model, genai.GenerativeModel):
            response = model.generate_content(prompt)
            if response.text:
                return response.text
            else:
                return "Gemini returned an empty response."
        else:
            return "The model was not correctly initialized."
    except Exception as e:
        return f"An error occurred during response generation: {e}"

def run_rag_query(query, vectorstore, api_key):
    """Run the RAG query by retrieving relevant chunks and generating a response."""
    try:
        model = setup_model(api_key)
        relevant_chunks = retrieve_relevant_chunks(vectorstore, query)
        prompt = generate_prompt(relevant_chunks, query)
        response = generate_response(model, prompt)
        return response
    except Exception as e:
        return f"Error running RAG query: {e}"

# --- web_app.py code (streamlit_app.py) ---

# Initialize session state variables
if "summary" not in st.session_state:
    st.session_state.summary = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "output_path" not in st.session_state:
    st.session_state.output_path = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Streamlit UI
st.title("PDF Summarizer & Question Answering")

# Input API key
api_key = st.text_input("Enter your Google API Key:", type="password")

if not api_key:
    st.warning("Please enter your API key to proceed.")
    st.stop()

pdf_source = st.radio("Select PDF Source:", ("The Gift of the Magi (Default)", "Upload Your Own PDF"))

uploaded_file = None
if pdf_source == "Upload Your Own PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

process_clicked = st.button("Process PDF")
if process_clicked:
    with st.spinner("Processing PDF... This may take a moment."):
        if pdf_source == "The Gift of the Magi (Default)":
            # Use PDF extraction and summarization
            st.session_state.output_path, summary_path = process_pdf(DEFAULT_PDF_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_SUMMARY_PATH, api_key)
            if st.session_state.output_path and summary_path:
                with open(DEFAULT_SUMMARY_PATH, 'r', encoding='utf-8') as f:
                    st.session_state.summary = f.read()
                st.session_state.pdf_filename = "The_Gift_of_the_Magi.pdf"
                # Create vectorstore
                st.session_state.vectorstore = create_vectorstore(DEFAULT_OUTPUT_PATH)
                st.success("PDF processed successfully!")

        elif uploaded_file is not None:
            # Save uploaded file
            filename = secure_filename(uploaded_file.name)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            upload_output_path = Path(OUTPUT_FOLDER) / f"{filename.split('.')[0]}_extracted.txt"
            upload_summary_path = Path(OUTPUT_FOLDER) / f"{filename.split('.')[0]}_summary.txt"

            st.session_state.output_path, summary_path = process_pdf(Path(filepath), upload_output_path, upload_summary_path, api_key)

            if st.session_state.output_path and summary_path:
                with open(upload_summary_path, 'r', encoding='utf-8') as f:
                    st.session_state.summary = f.read()
                st.session_state.pdf_filename = filename
                # Create vectorstore
                st.session_state.vectorstore = create_vectorstore(upload_output_path)
                st.success("PDF processed successfully!")
        else:
            st.error("Please upload a PDF file.")

# Display summary if available
if st.session_state.summary:
    st.subheader("Summary:")
    st.write(st.session_state.summary)

    # Check if vectorstore is created
    if st.session_state.vectorstore is None:
        st.warning("Vector database not initialized. Questions may not be answered accurately.")

    # Chat section
    st.subheader("Ask Questions About the Document")

    # Create a form for the question input
    with st.form(key="question_form"):
        question = st.text_input("Ask a Question:")
        submit_button = st.form_submit_button("Ask")

        if submit_button and question:
            with st.spinner("Generating answer..."):
                # Use RAG for answering questions
                answer = run_rag_query(question, st.session_state.vectorstore, api_key)

                if answer:
                    st.session_state.conversation_history.append({"question": question, "answer": answer})
                else:
                    st.error("Failed to get an answer. Please check the console for errors.")

    # Display conversation history outside the form
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for item in st.session_state.conversation_history:
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown("---")