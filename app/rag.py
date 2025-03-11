import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import google.generativeai as genai

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

# Function to create or load a Chroma vector store
def create_vectorstore(text_path):
    """Creates or loads a Chroma vectorstore from text chunks."""
    try:
        if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            # Load existing vectorstore
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            print("Loaded existing vectorstore.")
            return vectorstore

        # Load and chunk the text
        texts = load_and_chunk_text(text_path)
        if not texts:
            print("No text chunks created from the document.")
            return None

        # Create embeddings using HuggingFace and store them in Chroma vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
        print("Created new vectorstore.")
        return vectorstore
    except Exception as e:
        print(f"Error creating/loading vectorstore: {e}")
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

# If you need to manually set up the API key for Gemini
def setup_gemini_api(api_key):
    """Configure the Google Gemini API with the provided API key."""
    genai.configure(api_key=api_key)

#Example Usage
if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY") # Ensure GOOGLE_API_KEY is set in .env
    text_file_path = "your_text_file.txt" # replace with your file path
    query = "What is the main topic?"

    vectorstore = create_vectorstore(text_file_path)

    if vectorstore:
        response = run_rag_query(query, vectorstore, api_key)
        print("Response:", response)
    else:
        print("Vectorstore creation failed.")