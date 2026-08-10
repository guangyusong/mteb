from io import BytesIO

import pytest
from datasets import Dataset
from PIL import Image as PILImage

from mteb.tasks.retrieval.eng import mm_bright_retrieval as mm_bright


def _png_bytes() -> bytes:
    output = BytesIO()
    PILImage.new("RGB", (2, 3), color="red").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def mm_bright_data(monkeypatch):
    document_id = "abcd1234_0"
    image_id = "academia_abcd1234_figure.png"
    query_image_id = "academia_query.png"
    positive_images = [
        {"image_path": image_id, "source_passage_id": document_id}
    ]
    datasets = {
        "documents": Dataset.from_list(
            [
                {"id": document_id, "content": "relevant passage"},
                {"id": "q1_deadbeefdeadbeef_1", "content": "hard negative"},
            ]
        ),
        "examples": Dataset.from_list(
            [{"id": "q-text", "query": "question", "gold_ids": [document_id]}]
        ),
        "examples_multimodal": Dataset.from_list(
            [
                {
                    "id": "q-mm",
                    "query": "question with image",
                    "image_paths": [query_image_id],
                    "gold_ids": [document_id],
                    "positive_images": positive_images,
                    "negative_images": [],
                },
            ]
        ),
        "examples_images": Dataset.from_list(
            [{"path": query_image_id, "bytes": _png_bytes()}]
        ),
        "document_images": Dataset.from_list(
            [
                {"path": image_id, "bytes": _png_bytes()},
                {"path": "academia_bad_svg.jpg", "bytes": b"<svg></svg>"},
            ]
        ),
    }

    def load_parquet(config, domain):
        assert domain == "academia"
        return datasets[config]

    monkeypatch.setattr(mm_bright, "_load_parquet", load_parquet)


@pytest.mark.parametrize("variant", ["t2t", "it2t", "it2i", "it2it"])
def test_load_domain(mm_bright_data, variant):
    data = mm_bright._load_domain("academia", variant)
    corpus_ids = set(data["corpus"]["id"])
    query_ids = set(data["queries"]["id"])

    assert query_ids == set(data["relevant_docs"])
    assert all(
        document_id in corpus_ids
        for relevant in data["relevant_docs"].values()
        for document_id in relevant
    )

    if variant == "t2t":
        assert query_ids == {"q-text"}
    else:
        assert query_ids == {"q-mm"}

    if variant == "it2i":
        assert corpus_ids == {"academia_abcd1234_figure.png"}
    elif variant == "it2it":
        assert data["relevant_docs"]["q-mm"] == {
            mm_bright._pair_id("abcd1234_0", mm_bright.NO_IMAGE): 1,
            mm_bright._pair_id(
                "abcd1234_0", "academia_abcd1234_figure.png"
            ): 2,
        }
