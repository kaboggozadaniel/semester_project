import PyPDF2
import docx
from fastapi import UploadFile

def extract_text_from_file(file: UploadFile):
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    elif filename.endswith(".docx"):
        doc = docx.Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".txt"):
        return file.file.read().decode("utf-8", errors="ignore")

    else:
        return None