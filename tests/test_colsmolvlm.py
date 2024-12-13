from typing import Generator

import pytest
from colpali_engine.models import ColIdefics3
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

from byaldi import RAGMultiModalModel
from byaldi.colpali import ColPaliModel


@pytest.fixture(scope="module")
def colsmolvlm_rag_model() -> Generator[RAGMultiModalModel, None, None]:
    device = get_torch_device("auto")
    print(f"Using device: {device}")
    yield RAGMultiModalModel.from_pretrained("vidore/colsmolvlm-alpha", device=device)
    tear_down_torch()


@pytest.mark.slow
def test_load_colsmolvlm_from_pretrained(colsmolvlm_rag_model: RAGMultiModalModel):
    assert isinstance(colsmolvlm_rag_model, RAGMultiModalModel)
    assert isinstance(colsmolvlm_rag_model.model, ColPaliModel)
    assert isinstance(colsmolvlm_rag_model.model.model, ColIdefics3)
