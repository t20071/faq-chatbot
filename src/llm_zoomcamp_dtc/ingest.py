"""Download the DataTalksClub FAQ and build an in-memory minsearch index."""

import requests
from minsearch import Index

from llm_zoomcamp_dtc.config import FAQ_URL

TEXT_FIELDS = ["question", "section", "answer"]
KEYWORD_FIELDS = ["course"]


def load_faq_data(courses: list[str] | None = None) -> list[dict]:
    """Return FAQ records for all courses, or only for the given course ids."""
    response = requests.get(f"{FAQ_URL}/json/courses.json", timeout=30)
    response.raise_for_status()

    documents = []
    for course in response.json():
        if courses and course["course"] not in courses:
            continue
        course_response = requests.get(f"{FAQ_URL}{course['path']}", timeout=30)
        course_response.raise_for_status()
        documents.extend(course_response.json())

    return documents


def build_index(documents: list[dict]) -> Index:
    index = Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
    index.fit(documents)
    return index
