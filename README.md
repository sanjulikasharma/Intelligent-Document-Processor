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

## How Text is Extracted from the PDFs?
PDF is first converted into a base64 encoded string as Gemini API accepts base64 encoded data for PDF Processing. Then, Google Gemini API is utilized to perform OCR on the encoded PDF data.
To evaluate the text extraction, I created a different .txt file containing the story, and computed the accuracy using Character Error Rate (CER). 
**The CER score was 0.029, that means that the OCR was performed well.**

## How Summarization is Performed?
Summarization of the extracted text is performed using Google Gemini through the summarization.py module. The prompt instructed the Gemini to generate a summary with the specifics.

## Question Answering from PDFs with Gemini, Sentence Transformer, ChromaDB and Langchain
This Streamlit application enables users to ask questions about PDF documents and receive real-time answers. It leverages Google Gemini's powerful OCR and natural language processing capabilities, combined with Retrieval Augmented Generation (RAG), to provide accurate and contextually relevant responses.








