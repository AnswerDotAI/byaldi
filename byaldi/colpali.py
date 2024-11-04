import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import srsly
import torch
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from PIL import Image

from byaldi.objects import Result

# Import version directly from the package metadata
VERSION = version("Byaldi")


class ColPaliModel:
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        index_name: Optional[str] = None,
        verbose: int = 1,
        load_from_index: bool = False,
        index_root: str = ".byaldi",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        if isinstance(pretrained_model_name_or_path, Path):
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if (
            "colpali" not in pretrained_model_name_or_path.lower()
            and "colqwen2" not in pretrained_model_name_or_path.lower()
        ):
            raise ValueError(
                "This pre-release version of Byaldi only supports ColPali and ColQwen2 for now. Incorrect model name specified."
            )

        if verbose > 0:
            print(
                f"Verbosity is set to {verbose} ({'active' if verbose == 1 else 'loud'}). Pass verbose=0 to make quieter."
            )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_name = self.pretrained_model_name_or_path
        self.n_gpu = torch.cuda.device_count() if n_gpu == -1 else n_gpu
        device = (
            device or (
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        self.index_name = index_name
        self.verbose = verbose
        self.load_from_index = load_from_index
        self.index_root = index_root
        self.kwargs = kwargs
        self.collection = {}
        self.indexed_embeddings = []
        self.embed_id_to_doc_id = {}
        self.doc_id_to_metadata = {}
        self.doc_ids_to_file_names = {}
        self.doc_ids = set()

        if "colpali" in pretrained_model_name_or_path.lower():
            self.model = ColPali.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=(
                    "cuda"
                    if device == "cuda"
                    or (isinstance(device, torch.device) and device.type == "cuda")
                    else None
                ),
                token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=(
                    "cuda"
                    if device == "cuda"
                    or (isinstance(device, torch.device) and device.type == "cuda")
                    else None
                ),
                token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        self.model = self.model.eval()

        if "colpali" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColPaliProcessor,
                ColPaliProcessor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColQwen2Processor,
                ColQwen2Processor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )

        self.device = device
        if device != "cuda" and not (
            isinstance(device, torch.device) and device.type == "cuda"
        ):
            self.model = self.model.to(device)

        if not load_from_index:
            self.full_document_collection = False
            self.highest_doc_id = -1
        else:
            if self.index_name is None:
                raise ValueError("No index name specified. Cannot load from index.")

            index_path = Path(index_root) / Path(self.index_name)
            index_config = srsly.read_gzip_json(index_path / "index_config.json.gz")
            self.full_document_collection = index_config.get(
                "full_document_collection", False
            )
            self.resize_stored_images = index_config.get("resize_stored_images", False)
            self.max_image_width = index_config.get("max_image_width", None)
            self.max_image_height = index_config.get("max_image_height", None)

            if self.full_document_collection:
                collection_path = index_path / "collection"
                json_files = sorted(
                    collection_path.glob("*.json.gz"),
                    key=lambda x: int(x.stem.split(".")[0]),
                )

                for json_file in json_files:
                    loaded_data = srsly.read_gzip_json(json_file)
                    self.collection.update({int(k): v for k, v in loaded_data.items()})

                if self.verbose > 0:
                    print(
                        "You are using in-memory collection. This means every image is stored in memory."
                    )
                    print(
                        "You might want to rethink this if you have a large collection!"
                    )
                    print(
                        f"Loaded {len(self.collection)} images from {len(json_files)} JSON files."
                    )

            embeddings_path = index_path / "embeddings"
            embedding_files = sorted(
                embeddings_path.glob("embeddings_*.pt"),
                key=lambda x: int(x.stem.split("_")[1]),
            )
            self.indexed_embeddings = []
            for file in embedding_files:
                self.indexed_embeddings.extend(torch.load(file))

            self.embed_id_to_doc_id = srsly.read_gzip_json(
                index_path / "embed_id_to_doc_id.json.gz"
            )
            # Restore keys to integers
            self.embed_id_to_doc_id = {
                int(k): v for k, v in self.embed_id_to_doc_id.items()
            }
            self.highest_doc_id = max(
                int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values()
            )
            self.doc_ids = set(
                int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values()
            )
            try:
                # We don't want this error out with indexes created prior to 0.0.2
                self.doc_ids_to_file_names = srsly.read_gzip_json(
                    index_path / "doc_ids_to_file_names.json.gz"
                )
                self.doc_ids_to_file_names = {
                    int(k): v for k, v in self.doc_ids_to_file_names.items()
                }
            except FileNotFoundError:
                pass

            # Load metadata
            metadata_path = index_path / "metadata.json.gz"
            if metadata_path.exists():
                self.doc_id_to_metadata = srsly.read_gzip_json(metadata_path)
                # Convert metadata keys to integers
                self.doc_id_to_metadata = {
                    int(k): v for k, v in self.doc_id_to_metadata.items()
                }
            else:
                self.doc_id_to_metadata = {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = ".byaldi",
        **kwargs,
    ):
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            n_gpu=n_gpu,
            verbose=verbose,
            load_from_index=False,
            index_root=index_root,
            device=device,
            **kwargs,
        )

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = ".byaldi",
        **kwargs,
    ):
        index_path = Path(index_root) / Path(index_path)
        index_config = srsly.read_gzip_json(index_path / "index_config.json.gz")

        instance = cls(
            pretrained_model_name_or_path=index_config["model_name"],
            n_gpu=n_gpu,
            index_name=index_path.name,
            verbose=verbose,
            load_from_index=True,
            index_root=str(index_path.parent),
            device=device,
            **kwargs,
        )

        return instance

    def _export_index(self):
        if self.index_name is None:
            raise ValueError("No index name specified. Cannot export.")

        index_path = Path(self.index_root) / Path(self.index_name)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_path = index_path / "embeddings"
        embeddings_path.mkdir(exist_ok=True)
        num_embeddings = len(self.indexed_embeddings)
        chunk_size = 500
        for i in range(0, num_embeddings, chunk_size):
            chunk = self.indexed_embeddings[i : i + chunk_size]
            torch.save(chunk, embeddings_path / f"embeddings_{i}.pt")

        # Save index config
        index_config = {
            "model_name": self.model_name,
            "full_document_collection": self.full_document_collection,
            "highest_doc_id": self.highest_doc_id,
            "resize_stored_images": (
                True if self.max_image_width and self.max_image_height else False
            ),
            "max_image_width": self.max_image_width,
            "max_image_height": self.max_image_height,
            "library_version": VERSION,
        }
        srsly.write_gzip_json(index_path / "index_config.json.gz", index_config)

        # Save embed_id_to_doc_id mapping
        srsly.write_gzip_json(
            index_path / "embed_id_to_doc_id.json.gz", self.embed_id_to_doc_id
        )

        # Save doc_ids_to_file_names
        srsly.write_gzip_json(
            index_path / "doc_ids_to_file_names.json.gz", self.doc_ids_to_file_names
        )

        # Save metadata
        srsly.write_gzip_json(index_path / "metadata.json.gz", self.doc_id_to_metadata)

        # Save collection if using in-memory collection
        if self.full_document_collection:
            collection_path = index_path / "collection"
            collection_path.mkdir(exist_ok=True)
            for i in range(0, len(self.collection), 500):
                chunk = dict(list(self.collection.items())[i : i + 500])
                srsly.write_gzip_json(collection_path / f"{i}.json.gz", chunk)

        if self.verbose > 0:
            print(f"Index exported to {index_path}")

    def index(
        self,
        input_path: Union[str, Path],
        index_name: Optional[str] = None,
        doc_ids: Optional[List[int]] = None,
        store_collection_with_index: bool = False,
        overwrite: bool = False,
        metadata: Optional[List[Dict[str, Union[str, int]]]] = None,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
    ) -> Dict[int, str]:
        if (
            self.index_name is not None
            and (index_name is None or self.index_name == index_name)
            and not overwrite
        ):
            raise ValueError(
                f"An index named {self.index_name} is already loaded.",
                "Use add_to_index() to add to it or search() to query it.",
                "Pass a new index_name to create a new index.",
                "Exiting indexing without doing anything...",
            )
            return None
        if index_name is None:
            raise ValueError("index_name must be specified to create a new index.")
        if store_collection_with_index:
            self.full_document_collection = True

        index_path = Path(self.index_root) / Path(index_name)
        if index_path.exists():
            if overwrite is False:
                raise ValueError(
                    f"An index named {index_name} already exists.",
                    "Use overwrite=True to delete the existing index and build a new one.",
                    "Exiting indexing without doing anything...",
                )
                return None
            else:
                print(
                    f"overwrite is on. Deleting existing index {index_name} to build a new one."
                )
                shutil.rmtree(index_path)

        self.index_name = index_name
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height

        input_path = Path(input_path)
        if not hasattr(self, "highest_doc_id") or overwrite is True:
            self.highest_doc_id = -1

        if input_path.is_dir():
            items = list(input_path.iterdir())
            if doc_ids is not None and len(doc_ids) != len(items):
                raise ValueError(
                    f"Number of doc_ids ({len(doc_ids)}) does not match number of documents ({len(items)})"
                )
            if metadata is not None and len(metadata) != len(items):
                raise ValueError(
                    f"Number of metadata entries ({len(metadata)}) does not match number of documents ({len(items)})"
                )
            for i, item in enumerate(items):
                print(f"Indexing file: {item}")
                doc_id = doc_ids[i] if doc_ids else self.highest_doc_id + 1
                doc_metadata = metadata[doc_id] if metadata else None
                self.add_to_index(
                    item,
                    store_collection_with_index,
                    doc_id=doc_id,
                    metadata=doc_metadata,
                )
                self.doc_ids_to_file_names[doc_id] = str(item)
        else:
            if metadata is not None and len(metadata) != 1:
                raise ValueError(
                    "For a single document, metadata should be a list with one dictionary"
                )
            doc_id = doc_ids[0] if doc_ids else self.highest_doc_id + 1
            doc_metadata = metadata[0] if metadata else None
            self.add_to_index(
                input_path,
                store_collection_with_index,
                doc_id=doc_id,
                metadata=doc_metadata,
            )
            self.doc_ids_to_file_names[doc_id] = str(input_path)

        self._export_index()
        return self.doc_ids_to_file_names

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        store_collection_with_index: bool,
        doc_id: Optional[Union[int, List[int]]] = None,
        metadata: Optional[List[Dict[str, Union[str, int]]]] = None,
    ) -> Dict[int, str]:
        if self.index_name is None:
            raise ValueError(
                "No index loaded. Use index() to create or load an index first."
            )
        if not hasattr(self, "highest_doc_id"):
            self.highest_doc_id = -1
        # Convert single inputs to lists for uniform processing
        if isinstance(input_item, (str, Path)) and Path(input_item).is_dir():
            input_items = list(Path(input_item).iterdir())
        else:
            input_items = (
                [input_item] if not isinstance(input_item, list) else input_item
            )

        doc_ids = (
            [doc_id]
            if isinstance(doc_id, int)
            else (doc_id if doc_id is not None else None)
        )

        # Validate input lengths
        if doc_ids and len(doc_ids) != len(input_items):
            raise ValueError(
                f"Number of doc_ids ({len(doc_ids)}) does not match number of input items ({len(input_items)})"
            )

        # Process each input item
        for i, item in enumerate(input_items):
            current_doc_id = doc_ids[i] if doc_ids else self.highest_doc_id + 1 + i
            current_metadata = metadata if metadata else None

            if current_doc_id in self.doc_ids:
                raise ValueError(
                    f"Document ID {current_doc_id} already exists in the index"
                )

            self.highest_doc_id = max(self.highest_doc_id, current_doc_id)

            if isinstance(item, (str, Path)):
                item_path = Path(item)
                if item_path.is_dir():
                    self._process_directory(
                        item_path,
                        store_collection_with_index,
                        current_doc_id,
                        current_metadata,
                    )
                else:
                    self._process_and_add_to_index(
                        item_path,
                        store_collection_with_index,
                        current_doc_id,
                        current_metadata,
                    )
                self.doc_ids_to_file_names[current_doc_id] = str(item_path)
            elif isinstance(item, Image.Image):
                self._process_and_add_to_index(
                    item, store_collection_with_index, current_doc_id, current_metadata
                )
                self.doc_ids_to_file_names[current_doc_id] = "In-memory Image"
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")

        self._export_index()
        return self.doc_ids_to_file_names

    def _process_directory(
        self,
        directory: Path,
        store_collection_with_index: bool,
        base_doc_id: int,
        metadata: Optional[Dict[str, Union[str, int]]],
    ):
        for i, item in enumerate(directory.iterdir()):
            print(f"Indexing file: {item}")
            current_doc_id = base_doc_id + i
            self._process_and_add_to_index(
                item, store_collection_with_index, current_doc_id, metadata
            )
            self.doc_ids_to_file_names[current_doc_id] = str(item)

    def _process_and_add_to_index(
        self,
        item: Union[Path, Image.Image],
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        """TODO: THERE ARE TOO MANY FUNCTIONS DOING THINGS HERE. I blame Claude, but this is temporary anyway."""
        if isinstance(item, Path):
            if item.suffix.lower() == ".pdf":
                with tempfile.TemporaryDirectory() as path:
                    images = convert_from_path(
                        item,
                        thread_count=os.cpu_count() - 1,
                        output_folder=path,
                        paths_only=True,
                    )
                    for i, image_path in enumerate(images):
                        image = Image.open(image_path)
                        self._add_to_index(
                            image,
                            store_collection_with_index,
                            doc_id,
                            page_id=i + 1,
                            metadata=metadata,
                        )
            elif item.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image = Image.open(item)
                self._add_to_index(
                    image, store_collection_with_index, doc_id, metadata=metadata
                )
            else:
                raise ValueError(f"Unsupported input type: {item.suffix}")
        elif isinstance(item, Image.Image):
            self._add_to_index(
                item, store_collection_with_index, doc_id, metadata=metadata
            )
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")

    def _add_to_index(
        self,
        image: Image.Image,
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        page_id: int = 1,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        if any(
            entry["doc_id"] == doc_id and entry["page_id"] == page_id
            for entry in self.embed_id_to_doc_id.values()
        ):
            raise ValueError(
                f"Document ID {doc_id} with page ID {page_id} already exists in the index"
            )

        processed_image = self.processor.process_images([image])

        # Generate embedding
        with torch.inference_mode():
            processed_image = {k: v.to(self.device).to(self.model.dtype) for k, v in processed_image.items()}
            embedding = self.model(**processed_image)

        # Add to index
        embed_id = len(self.indexed_embeddings)
        self.indexed_embeddings.extend(list(torch.unbind(embedding.to("cpu"))))
        self.embed_id_to_doc_id[embed_id] = {"doc_id": doc_id, "page_id": int(page_id)}

        # Update highest_doc_id
        self.highest_doc_id = max(
            self.highest_doc_id,
            int(doc_id) if isinstance(doc_id, int) else self.highest_doc_id,
        )

        if store_collection_with_index:
            import base64
            import io

            # Resize image while maintaining aspect ratio
            if self.max_image_width and self.max_image_height:
                img_width, img_height = image.size
                aspect_ratio = img_width / img_height
                if img_width > self.max_image_width:
                    new_width = self.max_image_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_width = img_width
                    new_height = img_height
                if new_height > self.max_image_height:
                    new_height = self.max_image_height
                    new_width = int(new_height * aspect_ratio)
                if self.verbose > 2:
                    print(
                        f"Resizing image to {new_width}x{new_height}",
                        f"(aspect ratio {aspect_ratio:.2f}, original size {img_width}x{img_height},"
                        f"compression {new_width/img_width * new_height/img_height:.2f})",
                    )
                image = image.resize((new_width, new_height), Image.LANCZOS)

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            self.collection[int(embed_id)] = img_str

        # Add metadata
        if metadata:
            self.doc_id_to_metadata[doc_id] = metadata

        if self.verbose > 0:
            print(f"Added page {page_id} of document {doc_id} to index.")

    def remove_from_index(self):
        raise NotImplementedError("This method is not implemented yet.")

    def filter_embeddings(self,filter_metadata:Dict[str,str]):
        req_doc_ids = []
        for idx,metadata_dict in self.doc_id_to_metadata.items():
            for metadata_key,metadata_value in metadata_dict.items():
                if metadata_key in filter_metadata:
                    if filter_metadata[metadata_key] == metadata_value:
                        req_doc_ids.append(idx)
                        
        req_embedding_ids = [eid for eid,doc in self.embed_id_to_doc_id.items() if doc['doc_id'] in req_doc_ids]
        req_embeddings = [ie for idx,ie in enumerate(self.indexed_embeddings) if idx in req_embedding_ids]

        return req_embeddings, req_embedding_ids
    
    def search(
        self,
        query: Union[str, List[str]],
        k: int = 10,
        filter_metadata: Optional[Dict[str,str]] = None,
        return_base64_results: Optional[bool] = None,
    ) -> Union[List[Result], List[List[Result]]]:
        # Set default value for return_base64_results if not provided
        if return_base64_results is None:
            return_base64_results = bool(self.collection)

        valid_metadata_keys = list(self.doc_id_to_metadata.values())
        # Ensure k is not larger than the number of indexed documents
        k = min(k, len(self.indexed_embeddings))

        # Process query/queries
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query

        results = []
        for q in queries:
            # Process query
            with torch.inference_mode():
                batch_query = self.processor.process_queries([q])
                batch_query = {k: v.to(self.device).to(self.model.dtype) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            qs = list(torch.unbind(embeddings_query.to("cpu")))
            if not filter_metadata:
                req_embeddings = self.indexed_embeddings
            else:
                req_embeddings, req_embedding_ids = self.filter_embeddings(filter_metadata=filter_metadata) 
            # Compute scores
            scores = self.processor.score(qs,req_embeddings).cpu().numpy()

            # Get top k relevant pages
            top_pages = scores.argsort(axis=1)[0][-k:][::-1].tolist()

            # Create Result objects
            query_results = []
            for embed_id in top_pages:
                if filter_metadata:
                    adjusted_embed_id = req_embedding_ids[embed_id]
                else:
                    adjusted_embed_id = int(embed_id)
                doc_info = self.embed_id_to_doc_id[adjusted_embed_id]
                result = Result(
                    doc_id=doc_info["doc_id"],
                    page_num=int(doc_info["page_id"]),
                    score=float(scores[0][int(embed_id)]),
                    metadata=self.doc_id_to_metadata.get(int(doc_info["doc_id"]), {}),
                    base64=self.collection.get(adjusted_embed_id)
                    if return_base64_results
                    else None,
                )
                query_results.append(result)

            results.append(query_results)

        return results[0] if isinstance(query, str) else results

    def encode_image(
        self, input_data: Union[str, Image.Image, List[Union[str, Image.Image]]]
    ) -> torch.Tensor:
        """
        Compute embeddings for one or more images, PDFs, folders, or image files.

        Args:
            input_data (Union[str, Image.Image, List[Union[str, Image.Image]]]):
                A single image, PDF path, folder path, image file path, or a list of these.

        Returns:
            torch.Tensor: The computed embeddings for the input data.
        """
        if not isinstance(input_data, list):
            input_data = [input_data]

        images = []
        for item in input_data:
            if isinstance(item, Image.Image):
                images.append(item)
            elif isinstance(item, str):
                if os.path.isdir(item):
                    # Process folder
                    for file in os.listdir(item):
                        if file.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                        ):
                            images.append(Image.open(os.path.join(item, file)))
                elif item.lower().endswith(".pdf"):
                    # Process PDF
                    with tempfile.TemporaryDirectory() as path:
                        pdf_images = convert_from_path(
                            item, thread_count=os.cpu_count() - 1, output_folder=path
                        )
                        images.extend(pdf_images)
                elif item.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                ):
                    # Process image file
                    images.append(Image.open(item))
                else:
                    raise ValueError(f"Unsupported file type: {item}")
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")

        with torch.inference_mode():
            batch = self.processor.process_images(images)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            embeddings = self.model(**batch)

        return embeddings.cpu()

    def encode_query(self, query: Union[str, List[str]]) -> torch.Tensor:
        """
        Compute embeddings for one or more text queries.

        Args:
            query (Union[str, List[str]]):
                A single text query or a list of text queries.

        Returns:
            torch.Tensor: The computed embeddings for the input query/queries.
        """
        if isinstance(query, str):
            query = [query]

        with torch.inference_mode():
            batch = self.processor.process_queries(query)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            embeddings = self.model(**batch)

        return embeddings.cpu()

    def get_doc_ids_to_file_names(self):
        return self.doc_ids_to_file_names
