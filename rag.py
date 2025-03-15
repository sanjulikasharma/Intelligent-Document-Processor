import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from tempfile import NamedTemporaryFile

# Directory to persist the Chroma vector store (use a Streamlit cache directory)
PERSIST_DIRECTORY = st.session_state.get("persist_directory", "chroma_db")

# Function to load and chunk text from a file (using Streamlit file uploader)
def load_and_chunk_text(uploaded_file):
    """Load text from uploaded file and chunk it into smaller pieces."""
    if uploaded_file is None:
        return []

    try:
        text = uploaded_file.getvalue().decode("utf-8")

        # Chunk text into smaller parts using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        return texts

    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        return []

# Function to create or load a Chroma vector store (using Streamlit cache)
def create_vectorstore(uploaded_file):
    """Creates or loads a Chroma vectorstore from text chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if "vectorstore" in st.session_state and os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            # Load existing vectorstore
            vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            st.write("Loaded existing vectorstore.")
            return vectorstore

        # Load and chunk the text
        texts = load_and_chunk_text(uploaded_file)
        if not texts:
            st.write("No text chunks created from the document.")
            return None

        # Create embeddings using HuggingFace and store them in Chroma vectorstore
        vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
        st.session_state["vectorstore"] = vectorstore #Store the vectorstore in streamlit's session_state.
        st.write("Created new vectorstore.")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating/loading vectorstore: {e}")
        return None

# Function to retrieve relevant chunks from the vectorstore based on a query
def retrieve_relevant_chunks(vectorstore, query):
    """Retrieve relevant chunks from the vectorstore based on the query."""
    if not vectorstore:
        return []

    try:
        # Retrieve relevant chunks by performing a similarity search in the vectorstore
        relevant_chunks = vectorstore.similarity_search(query)
        return relevant_chunks
    except Exception as e:
        st.error(f"Error in retrieving relevant chunks: {e}")
        return []

# Function to generate the RAG prompt based on the relevant chunks
def generate_prompt(relevant_chunks, query):
    """Generate a prompt using the relevant chunks of text and the user's query."""
    if not relevant_chunks:
        return "No relevant information found in the document."

    # Concatenate the content of the relevant chunks to create context for the prompt
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    # Simple Prompt added here.
    prompt = f"""
    You are a helpful chatbot that answers questions based on the provided document context.
    Context: {context}
    Question: {query}
    Answer:
    """
    return prompt

# Set up the Gemini model
def setup_model(api_key):
    """Configure API key for Gemini and initialize the model."""
    if not api_key:
        raise ValueError("Gemini API key is missing.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

# Function to generate a response using the Gemini model
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

# Main RAG function that runs the query, retrieves relevant chunks, and generates a response
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

# Streamlit UI
st.title("Document Q&A with Gemini")

api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
query = st.text_input("Enter your question:")

if uploaded_file and api_key and query:
    vectorstore = create_vectorstore(uploaded_file)
    if vectorstore:
        if st.button("Get Answer"):
            response = run_rag_query(query, vectorstore, api_key)
            st.write("Response:", response)