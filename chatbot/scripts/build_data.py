"""Download the DataTalksClub FAQ into data/ so the deployed app needs no network calls at startup.

Run from the chatbot/ folder:  python scripts/build_data.py
"""

import json
from pathlib import Path

import requests

FAQ_URL = "https://datatalks.club/faq"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    courses = requests.get(f"{FAQ_URL}/json/courses.json", timeout=30).json()

    documents = []
    for course in courses:
        response = requests.get(f"{FAQ_URL}{course['path']}", timeout=30)
        response.raise_for_status()
        documents.extend(response.json())

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False), encoding="utf-8"
    )
    course_list = [{"id": c["course"], "name": c["course_name"]} for c in courses]
    courses_json = json.dumps(course_list, indent=1)
    (DATA_DIR / "courses.json").write_text(courses_json, encoding="utf-8")
    # Also served as a static file, so the page can show courses without a function call.
    (DATA_DIR.parent / "public" / "courses.json").write_text(courses_json, encoding="utf-8")

    print(f"Saved {len(documents)} documents from {len(courses)} courses to {DATA_DIR}")
    print("Now re-embed them: uv run python chatbot/scripts/build_embeddings.py")


if __name__ == "__main__":
    main()
