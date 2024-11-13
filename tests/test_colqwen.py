from typing import Generator

import pytest
from colpali_engine.models import ColQwen2
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

from byaldi import RAGMultiModalModel
from byaldi.colpali import ColPaliModel


@pytest.fixture(scope="module")
def colqwen_rag_model() -> Generator[RAGMultiModalModel, None, None]:
    device = get_torch_device("auto")
    print(f"Using device: {device}")
    yield RAGMultiModalModel.from_pretrained("vidore/colqwen2-v0.1", device=device)
    tear_down_torch()


@pytest.mark.slow
def test_load_colqwen_from_pretrained(colqwen_rag_model: RAGMultiModalModel):
    assert isinstance(colqwen_rag_model, RAGMultiModalModel)
    assert isinstance(colqwen_rag_model.model, ColPaliModel)
    assert isinstance(colqwen_rag_model.model.model, ColQwen2)
