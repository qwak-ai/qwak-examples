import PyPDF2
import sys


def pdf_to_text(pdf_file_path: str, text_file_path: str):

    # Open the PDF file in read-binary mode
    with open(pdf_file_path, 'rb') as pdf_file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Initialize an empty string to hold the extracted text
        extracted_text = ''

        # Loop through each page in the PDF and extract the text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extract_text()

    # Write the extracted text to a .txt file
    with open(text_file_path, 'w') as text_file:
        text_file.write(extracted_text)


if __name__ == '__main__':
    # Usage: python3 pdf_totext.py data/myfile.pdf
    # pdf_in = sys.argv[1]
    pdf_in = "data/Tesla-10k.pdf"
    txt_out = "data/Tesla-10k.txt"
    pdf_to_text(pdf_in, txt_out)
