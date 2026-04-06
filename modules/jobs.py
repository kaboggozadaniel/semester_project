import pandas as pd

# Load once 
job_df = pd.read_csv("data/jobs.csv")

def get_job_recommendations(category, top_n=3):
    jobs = job_df[job_df['Category'] == category]
    
    if jobs.empty:
        return []
    
    return jobs.head(top_n).to_dict(orient='records')