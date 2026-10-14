"""Persistent FAQ search index stored in database/faq.db.

Build or rebuild it from the command line:

    uv run build-faq-db                      # LLM Zoomcamp only (default)
    uv run build-faq-db --course all         # every course
"""

import argparse
from pathlib import Path

from sqlitesearch import TextSearchIndex

from llm_zoomcamp_dtc.config import DATABASE_PATH
from llm_zoomcamp_dtc.ingest import KEYWORD_FIELDS, TEXT_FIELDS, load_faq_data


def open_sqlite_index(db_path: Path = DATABASE_PATH) -> TextSearchIndex:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return TextSearchIndex(
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS,
        db_path=str(db_path),
    )


def build_sqlite_index(documents: list[dict], db_path: Path = DATABASE_PATH) -> TextSearchIndex:
    """Replace the index contents with `documents`."""
    index = open_sqlite_index(db_path)
    index.clear()
    index.fit(documents)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SQLite FAQ index in database/")
    parser.add_argument("--course", default="llm-zoomcamp", help='course id, or "all"')
    args = parser.parse_args()

    courses = None if args.course == "all" else [args.course]
    documents = load_faq_data(courses)
    index = build_sqlite_index(documents)
    print(f"Indexed {index.count()} documents into {DATABASE_PATH}")
    index.close()


if __name__ == "__main__":
    main()
