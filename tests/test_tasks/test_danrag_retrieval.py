from datasets import Dataset

from mteb.tasks.retrieval.dan import danrag_retrieval


def test_load_danrag_data_builds_visual_and_multimodal_corpora(
    monkeypatch,
) -> None:
    source_data = {
        "corpus/*.parquet": Dataset.from_dict(
            {
                "page_id": ["page-1", "page-2"],
                "text": ["first page", "second page"],
                "image": [None, None],
                "sector": ["legal", "legal"],
            }
        ),
        "queries/*.parquet": Dataset.from_dict(
            {
                "id": ["query-1", "query-2"],
                "query": ["first question", "second question"],
                "valid_pages": [["page-1", "page-2"], ["page-2"]],
                "answer": ["first answer", "second answer"],
            }
        ),
    }
    load_calls = []

    def mock_load_dataset(
        path: str,
        *,
        data_files: dict[str, str],
        split: str,
        revision: str,
        num_proc: int | None,
    ) -> Dataset:
        load_calls.append((path, data_files, split, revision, num_proc))
        return source_data[data_files[split]]

    monkeypatch.setattr(danrag_retrieval, "load_dataset", mock_load_dataset)

    visual = danrag_retrieval._load_danrag_data(
        path="source/repository",
        revision="revision",
        splits=["test"],
        include_text=False,
        num_proc=2,
    )
    multimodal = danrag_retrieval._load_danrag_data(
        path="source/repository",
        revision="revision",
        splits=["test"],
        include_text=True,
        num_proc=2,
    )

    assert len(load_calls) == 4
    assert load_calls[0][1] == {"test": "corpus/*.parquet"}
    assert load_calls[1][1] == {"test": "queries/*.parquet"}

    visual_split = visual["default"]["test"]
    assert visual_split["corpus"].column_names == ["id", "image"]
    assert visual_split["queries"].to_dict() == {
        "id": ["query-1", "query-2"],
        "text": ["first question", "second question"],
    }
    assert visual_split["relevant_docs"] == {
        "query-1": {"page-1": 1, "page-2": 1},
        "query-2": {"page-2": 1},
    }
    assert visual_split["top_ranked"] is None

    multimodal_split = multimodal["default"]["test"]
    assert multimodal_split["corpus"].column_names == ["id", "text", "image"]
