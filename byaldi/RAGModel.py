from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from byaldi.colpali import ColPaliModel
from byaldi.indexing import IndexManager
from byaldi.objects import Result


class RAGMultiModalModel:
    def __init__(
        self,
        model: Optional[ColPaliModel] = None,
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        self.model = model
        self.index_manager = IndexManager(index_root=index_root, verbose=verbose)
        self.device = device
        self.verbose = verbose

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        model = ColPaliModel.from_pretrained(
            pretrained_model_name_or_path,
            device=device,
            verbose=verbose,
        )
        return cls(model, index_root, device, verbose)

    @classmethod
    def from_index(
        cls,
        index_name: str,
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        instance = cls(index_root=index_root, device=device, verbose=verbose)
        instance.index_manager.load_index(index_name)
        instance.model = ColPaliModel.from_pretrained(
            instance.index_manager.model_name,
            device=device,
            verbose=verbose,
        )
        return instance

    def index(
        self,
        input_path: Union[str, Path],
        index_name: str,
        store_collection_with_index: bool = False,
        doc_ids: Optional[List[int]] = None,
        metadata: Optional[List[Dict[str, Union[str, int]]]] = None,
        overwrite: bool = False,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
        **kwargs,
    ):
        self.index_manager.create_index(
            index_name,
            store_collection_with_index,
            overwrite,
            max_image_width,
            max_image_height,
        )
        return self.index_manager.add_to_index(
            input_path,
            self.model.encode_image,
            doc_ids,
            metadata,
        )

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image],
        store_collection_with_index: bool = False,
        doc_id: Optional[int] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        return self.index_manager.add_to_index(
            input_item,
            self.model.encode_image,
            doc_id,
            metadata,
            store_collection_with_index,
        )

    def search(
        self,
        query: Union[str, List[str]],
        k: int = 10,
        return_base64_results: Optional[bool] = None,
    ) -> Union[List[Result], List[List[Result]]]:
        return self.index_manager.search(
            query,
            self.model.score,
            k,
            return_base64_results,
        )

    def get_doc_ids_to_file_names(self):
        return self.index_manager.get_doc_ids_to_file_names()

    def as_langchain_retriever(self, **kwargs: Any):
        from byaldi.integrations import ByaldiLangChainRetriever
        return ByaldiLangChainRetriever(model=self, kwargs=kwargs)