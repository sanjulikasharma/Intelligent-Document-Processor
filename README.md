# Intelligent Document Processor 

The Streamlit application allows users to upload or select a PDF document, extract text using Google Gemini's OCR capabilities, summarize its content, and ask questions about it using Retrieval Augmented Generation (RAG).

## How to run the application?

### Step 1: Clone the Repository 
Open your terminal or command prompt and run the following command:

    ``` bash
    git clone https://github.com/sanjulikasharma/Intelligent-Document-Processor.git
    ```
### Step 2: Install Dependencies using `requirements.txt`
    ``` bash
    pip install -r requirements.txt
    ```
### Step 3: Running the Application 
Navigate to the project directory in your terminal or command prompt (if you're not already there) and run the Streamlit application: 

    ```bash
    streamlit run web_app.py
    ```
This will start the Streamlit server, and your application will open in your default web browser.

**After entering the Google Gemini API key, you can upload your pdf or use the default 'The Gift of Magi' pdf to generate a summary & ask questions related to the pdf.'**
## Dependencies

-   Streamlit
-   LangChain
-   Hugging Face Transformers
-   ChromaDB
-   Google Generative AI
-   python-dotenv
-   Werkzeug
-   editdistance
-   pypdf

## Libraries/APIs 

-   **Streamlit:** Chosen for its simplicity in creating web applications with Python, allowing for rapid prototyping and deployment.
-   **LangChain:** Used for its powerful framework for developing applications powered by language models, specifically for text splitting and vector store interactions.
-   **Hugging Face Transformers:** Employed for embedding generation, leveraging the `all-MiniLM-L6-v2` model for efficient and effective text embeddings.
-   **ChromaDB:** Selected for its ease of use and persistence capabilities in storing and retrieving vector embeddings, enabling efficient similarity searches.
-   **Google Generative AI (Gemini):** Utilized for its state-of-the-art capabilities in OCR, summarization, and question answering, providing high-quality results.
-   **python-dotenv:** Used to manage environment variables, keeping sensitive information like API keys secure.
-   **Werkzeug:** Employed for secure file name handling during PDF uploads.

**Design Choices:**

-   **RAG Implementation:** The application implements a Retrieval Augmented Generation (RAG) approach to answer questions, ensuring that the responses are grounded in the provided document's content.
-   **Modular Design:** The code is structured into separate modules (`rag.py`, `pdf_to_text_extraction.py`, `summarization.py`) for better organization and maintainability.
-   **Error Handling:** Basic error handling is implemented to provide informative messages to the user in case of issues like missing files or API key errors.

## Notes

-   Ensure you have a valid Google Gemini API key.
-   The application uses ChromaDB to persist the vector store.
-   The `uploads` and `output` directories are created automatically if they don't exist.
-   The `chroma_db` folder is used to persist the vector database.
-   The `data` folder is used to store default pdfs.
-   The `pdf_to_text_extraction.py` file utilizes Google Gemini's multimodal capabilities to perform OCR on PDF documents by converting them to base64 and sending them to the Gemini API.
-   The `summarization.py` file uses Google Gemini to generate summaries of the extracted text.
-   The optional CER calculation in `pdf_to_text_extraction.py` allows for evaluating the accuracy of the OCR process.
-   **Python version 3.12 or greater is required for this application.**






