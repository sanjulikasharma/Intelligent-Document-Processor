import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv
from pdf_to_text_extraction import get_api_key  # Import get_api_key

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
        print(f"An error occurred during summarization: {e}")
        return None

def main():
    """Main function to demonstrate text summarization."""
    output_dir = Path("output")  # The output directory
    input_text_path = output_dir / "extracted_text.txt"  # Path to the extracted text.
    summary_path = output_dir / "summary.txt"  # Path to save summary.

    load_dotenv()  # Load the .env file

    try:
        api_key = get_api_key()  # Use get_api_key from pdf_to_text_extraction
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not input_text_path.exists():
        print(f"Error: Input text file not found at {input_text_path}")
        return

    try:
        with open(input_text_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()
    except Exception as e:
        print(f"Error reading input text file: {e}")
        return

    summary = summarize_text(extracted_text, api_key)

    if summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved to {summary_path}")
    else:
        print("Summarization failed.")

if __name__ == "__main__":
    main()