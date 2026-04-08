# Resume Analyzer & Career Recommendation System

## Overview

This is a semester project for the course **92400566001_SEM_PROJECT**. The system is an AI-powered resume analyzer that uses Natural Language Processing (NLP) and Machine Learning to classify resumes into job categories and provide personalized career recommendations.

## Features

- **Resume Classification**: Analyzes resume text using TF-IDF vectorization and Logistic Regression to predict job categories
- **Job Recommendations**: Provides relevant job suggestions based on the predicted category
- **Web Interface**: User-friendly web application built with FastAPI and Jinja2 templates
- **Real-time Analysis**: Instant results with confidence scores

## Technology Stack

- **Backend**: FastAPI (Python web framework)
- **Machine Learning**: Scikit-learn (TF-IDF, Logistic Regression)
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, Jinja2 templates
- **Deployment**: Uvicorn (ASGI server)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project files
2. Navigate to the project directory:
   ```bash
   cd semester_project
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the model files are present:
   - `model.pkl` (trained Logistic Regression model)
   - `vectorizer.pkl` (TF-IDF vectorizer)

## Usage

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Open your browser and navigate to `http://127.0.0.1:8000`

3. Paste your resume text into the textarea and click "Analyze"

### API Endpoints

- `GET /`: Home page with the resume analyzer form
- `POST /predict`: Analyzes the submitted resume and returns predictions with job recommendations

## Project Structure

```
semester_project/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── model.pkl              # Trained machine learning model
├── vectorizer.pkl         # TF-IDF vectorizer
├── data/
│   ├── jobs.csv           # Job recommendations database
│   └── Resume.csv         # Training dataset (resumes)
├── model/
│   └── model.py           # Model training script
├── modules/
│   ├── utils.py           # Text preprocessing utilities
│   └── jobs.py            # Job recommendation logic
└── templates/
    ├── index.html         # Main web interface
    └── index.css          # Styling
```

## Model Details

### Training Data
- Dataset: Resume.csv (contains resume text and corresponding job categories)
- Features: Cleaned and preprocessed resume text using TF-IDF vectorization
- Target: Job category classification

### Model Architecture
- **Vectorizer**: TF-IDF with n-grams (1-2), max features 12000, custom token pattern
- **Classifier**: Logistic Regression with balanced class weights
- **Performance**: Cross-validation accuracy ~85% (approximate)

### Preprocessing Steps
1. Text cleaning: Remove special characters, convert to lowercase
2. Resume trimming: Keep first and last 500 characters for long resumes
3. Vectorization: Convert text to numerical features

## Data Sources

- **Resume Dataset**: Used for training the classification model
- **Jobs Dataset**: Contains job listings categorized by type for recommendations