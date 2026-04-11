from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

import PyPDF2
import docx
import io

import pickle
import re

from modules.utils import clean_data, trim_resume
from modules.keyword_extract import extract_keywords
from modules.file_handler import extract_text_from_file
from modules.jobs import get_job_recommendations
from model.predict import predict_resume

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.cache = None

app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})
      

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, resume_text: Optional[str] = Form(None), file:UploadFile = File(None)):
    
    if file and file.filename:
        extracted_text = extract_text_from_file(file)
        if not extracted_text:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Unsupported file format"
            })
        resume_text = extracted_text

    if resume_text is None or not resume_text.strip():
        return templates.TemplateResponse(request, "index.html", {
            "request": request,
            "prediction": None,
            "confidence": None,
            "jobs": [],
            "error": "Please enter your resume details."
        })

    # Preprocess
    cleaned = clean_data(resume_text)
    if not cleaned or not cleaned.strip():
        return templates.TemplateResponse(request, "index.html", {
            "request": request,
            "prediction": None,
            "confidence": None,
            "jobs": [],
            "error": "Resume text does not contain any valid content. Please paste your resume text again."
        })
    trimmed = trim_resume(cleaned)

    # prediction
    prediction, confidence = predict_resume(trimmed)

    # job recommendations
    jobs = get_job_recommendations(prediction)
    keywords = extract_keywords(resume_text)
    
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "prediction": prediction,
        "confidence": f"{confidence}%",
        "jobs": jobs,
        "keywords": keywords
    })
