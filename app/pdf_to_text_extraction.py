import google.generativeai as genai
import os
import base64
from pathlib import Path
import editdistance
from dotenv import load_dotenv

load_dotenv()

API_KEY_ENV_VAR = "GOOGLE_API_KEY"

def get_api_key():

   #Retrieves the API key from environment variables.
    api_key = os.getenv(API_KEY_ENV_VAR)

    if not api_key:
        print(f"Error: API key not found. Please set the '{API_KEY_ENV_VAR}' environment variable in a .env file.")
        return None

    if not api_key:
        print("Error: API key is empty.")
        return None

    return api_key

def setup_gemini_api(api_key):
    genai.configure(api_key=api_key)

# Converts a PDF file to a base64 encoded string
def pdf_to_base64(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
        return pdf_data
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred during PDF to base64 conversion: {e}")
        return None

# Extracts text from PDF using Gemini 
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
        print(f"An error occurred during OCR: {e}")
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
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during CER calculation: {e}")
        return None
    

def main():
    """Main function to demonstrate OCR and CER calculation."""

    pdf_file_path = Path("D:\Intelligent Document Processor\data\The_Gift_of_the_Magi.pdf")
    ground_truth_file = Path("D:\Intelligent Document Processor\output\ground_truth.txt")
    output_path = Path("D:\Intelligent Document Processor\output\extracted_text.txt")

    api_key = get_api_key()

    if not api_key:
        print(f'Error: Enter the API Key')
        return 

    if not pdf_file_path.exists():
        print(f"Error: PDF file not found at {pdf_file_path}")
        return

    extracted_text = extract_text_from_pdf(pdf_file_path, api_key)

    if extracted_text:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"Text extracted and saved to {output_path}")

        if ground_truth_file.exists():
            cer_value = calculate_cer(ground_truth_file, output_path)
            if cer_value is not None:
                print(f"Character Error Rate (CER): {cer_value:.4f}")
        else:
            print(f"Ground truth file not found at {ground_truth_file}. CER calculation skipped.")

    else:
        print("OCR failed.")

if __name__ == "__main__":
    main()