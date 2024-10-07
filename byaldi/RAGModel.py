from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from byaldi.colpali import ColPaliModel
from byaldi.indexing import IndexManager
from byaldi.objects import Result


class RAGMultiModalModel:
    """
    Wrapper class for a pretrained RAG multi-modal model, and an associated index manager.
    Allows you to load a pretrained model from disk or from the hub, build or query an index.
    ## Usage
    Load a pre-trained checkpoint:
    ```python
    from byaldi import RAGMultiModalModel
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
    ```
    Both methods will load a fully initialised instance of ColPali, which you can use to build and query indexes.
    ```python
    RAG.search("How many people live in France?")
    ```
    """
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
        """
        Load a ColPali model from a pre-trained checkpoint.
        Parameters:
            pretrained_model_name_or_path (str): Local path or huggingface model name.
            device (str): The device to load the model on. Default is "cuda".
        Returns:
            cls (RAGMultiModalModel): The current instance of RAGMultiModalModel, with the model initialised.
        """
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
        """
        Load an Index and the associated ColPali model from an existing document index.

        Parameters:
            index_name (str): Name of the index.
            index_root (str): Path to the index root directory.
            device (str): The device to load the model on. Default is "cuda".
            verbose (int): Verbosity level.

        Returns:
            cls (RAGMultiModalModel): The current instance of RAGMultiModalModel, with the index and model loaded.
        """
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
        """
        Wrapper function to create and add to an index.

        Parameters:
            input_path (str, Path): Path to the input file or directory.
            index_name (str): Name of the index.
            store_collection_with_index (bool): Whether to store the collection with the index.
            doc_ids (List[int]): List of document IDs.
            metadata (List[Dict[str, Union[str, int]]]): List of metadata dictionaries.
            overwrite (bool): Whether to overwrite the existing index.
            max_image_width (int): Maximum image width.
            max_image_height (int): Maximum image height.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
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
        """
        Wrapper function to add to an existing index.

        Parameters:
            input_item (str, Path, Image.Image): Input file or directory.
            store_collection_with_index (bool): Whether to store the collection with the index.
            doc_id (int): Document ID.
            metadata (Dict[str, Union[str, int]]): Metadata dictionary.

        Returns:
            None
        """

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
        """
        Search the index for the given query.

        Parameters:
            query (str, List[str]): Query string or list of query strings.
            k (int): Number of results to return.
            return_base64_results (bool): Whether to return base64 encoded results.

        Returns:
            Union[List[Result], List[List[Result]]]: List of Result objects or list of lists of Result objects.
        """
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