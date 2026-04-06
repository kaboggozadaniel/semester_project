from typing import Optional

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import pickle
import re

from modules.utils import clean_data, trim_resume
from modules.jobs import get_job_recommendations

app = FastAPI()

# model 
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# template
templates = Jinja2Templates(directory="templates")
templates.env.cache = None

app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})
      

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, resume_text: Optional[str] = Form(None)):
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

    # Vectorize
    vectorized = vectorizer.transform([trimmed])

    # Predict
    prediction = model.predict(vectorized)[0]
    confidence = max(model.predict_proba(vectorized)[0])

    # job recommendations
    jobs = get_job_recommendations(prediction)

    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "prediction": prediction,
        "confidence": f"{confidence*100:.2f}%",
        "jobs": jobs
    })
