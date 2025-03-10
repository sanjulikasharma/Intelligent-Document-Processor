import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from langchain.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Directory to persist the Chroma vector store
PERSIST_DIRECTORY = "chroma_db"

# Function to load and chunk text from a file
def load_and_chunk_text(file_path):
    """Load text from file and chunk it into smaller pieces."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Chunk text into smaller parts using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        return texts
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return []

# Function to create a Chroma vector store
def create_vectorstore(text_path):
    """Creates a Chroma vectorstore from text chunks."""
    try:
        # Load and chunk the text
        texts = load_and_chunk_text(text_path)
        if not texts:
            print("No text chunks created from the document.")
            return None
            
        # Create embeddings using HuggingFace and store them in Chroma vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
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
        print(f"Error in retrieving relevant chunks: {e}")
        return []

# Function to generate the RAG prompt based on the relevant chunks
def generate_prompt(relevant_chunks, query):
    """Generate a prompt using the relevant chunks of text and the user's query."""
    if not relevant_chunks:
        return "No relevant information found in the document."
    
    # Concatenate the content of the relevant chunks to create context for the prompt
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    prompt = f"""
    Answer the question based on the context provided.
    Context: {context}
    Question: {query}
    Answer:
    """
    return prompt

import google.generativeai as genai

# Set up the Gemini model
def setup_model(api_key):
    # Configure API key for Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')  # Ensure this is correctly initialized as a model
    return model

# Function to generate a response using the Gemini model
def generate_response(model, prompt):
    """Generate a response using the Gemini model."""
    try:
        if isinstance(model, genai.GenerativeModel):  # Ensure it's the correct type
            response = model.generate_content(prompt)
            return response.text
        else:
            return "The model was not correctly initialized."
    except Exception as e:
        return f"An error occurred during response generation: {e}"


# Main RAG function that runs the query, retrieves relevant chunks, and generates a response
def run_rag_query(query, vectorstore, api_key):
    """Run the RAG query by retrieving relevant chunks and generating a response."""
    model = setup_model(api_key)  # Initialize the model with API key
    
    relevant_chunks = retrieve_relevant_chunks(vectorstore, query)
    prompt = generate_prompt(relevant_chunks, query)
    
    response = generate_response(model, prompt)
    return response

# If you need to manually set up the API key for Gemini 
def setup_gemini_api(api_key):
    """Configure the Google Gemini API with the provided API key."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)

