import pandas as pd

columns = [
    "title",
    "company",
    "location",
    "link",
    "source",
    "date_posted",
    "work_type",
    "employment_type",
]


job_df = pd.read_csv("data/jobs.csv")[columns].fillna("Not Specified")


def get_job_recommendations(category=None, top_n=3):
    jobs = job_df

    if category:
        title_match = job_df["title"].str.contains(str(category), case=False, na=False)
        if title_match.any():
            jobs = job_df[title_match]

    return jobs.head(top_n).to_dict(orient="records")
