import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import srsly
import torch
from pdf2image import convert_from_path
from PIL import Image

from byaldi.objects import Result


class IndexManager:
    def __init__(
        self,
        index_root: str = ".byaldi",
        verbose: int = 1,
    ):
        self.index_root = index_root
        self.verbose = verbose
        self.collection = {}
        self.indexed_embeddings = []
        self.embed_id_to_doc_id = {}
        self.doc_id_to_metadata = {}
        self.doc_ids_to_file_names = {}
        self.doc_ids = set()
        self.highest_doc_id = -1
        self.full_document_collection = False
        self.index_name = None
        self.max_image_width = None
        self.max_image_height = None

    def create_index(
        self,
        index_name: str,
        store_collection_with_index: bool = False,
        overwrite: bool = False,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
    ):
        if self.index_name is not None and not overwrite:
            raise ValueError(f"An index named {self.index_name} is already loaded.")
        
        index_path = Path(self.index_root) / Path(index_name)
        if index_path.exists():
            if not overwrite:
                raise ValueError(f"An index named {index_name} already exists.")
            else:
                shutil.rmtree(index_path)

        self.index_name = index_name
        self.full_document_collection = store_collection_with_index
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.highest_doc_id = -1

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        embed_func,
        doc_id: Optional[Union[int, List[int]]] = None,
        metadata: Optional[List[Dict[str, Union[str, int]]]] = None,
    ) -> Dict[int, str]:
        if self.index_name is None:
            raise ValueError("No index loaded. Use create_index() first.")

        input_items = [input_item] if not isinstance(input_item, list) else input_item
        doc_ids = [doc_id] if isinstance(doc_id, int) else (doc_id if doc_id is not None else None)

        for i, item in enumerate(input_items):
            current_doc_id = doc_ids[i] if doc_ids else self.highest_doc_id + 1 + i
            current_metadata = metadata[i] if metadata else None

            if current_doc_id in self.doc_ids:
                raise ValueError(f"Document ID {current_doc_id} already exists in the index")

            self.highest_doc_id = max(self.highest_doc_id, current_doc_id)

            if isinstance(item, (str, Path)):
                item_path = Path(item)
                if item_path.is_dir():
                    self._process_directory(item_path, embed_func, current_doc_id, current_metadata)
                else:
                    self._process_and_add_to_index(item_path, embed_func, current_doc_id, current_metadata)
                self.doc_ids_to_file_names[current_doc_id] = str(item_path)
            elif isinstance(item, Image.Image):
                self._process_and_add_to_index(item, embed_func, current_doc_id, current_metadata)
                self.doc_ids_to_file_names[current_doc_id] = "In-memory Image"
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")

        self._export_index()
        return self.doc_ids_to_file_names

    def _process_directory(
        self,
        directory: Path,
        embed_func,
        base_doc_id: int,
        metadata: Optional[Dict[str, Union[str, int]]],
    ):
        for i, item in enumerate(directory.iterdir()):
            if self.verbose > 0:
                print(f"Indexing file: {item}")
            current_doc_id = base_doc_id + i
            self._process_and_add_to_index(item, embed_func, current_doc_id, metadata)
            self.doc_ids_to_file_names[current_doc_id] = str(item)

    def _process_and_add_to_index(
        self,
        item: Union[Path, Image.Image],
        embed_func,
        doc_id: Union[str, int],
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        if isinstance(item, Path):
            if item.suffix.lower() == ".pdf":
                with tempfile.TemporaryDirectory() as path:
                    images = convert_from_path(item, thread_count=os.cpu_count() - 1, output_folder=path, paths_only=True)
                    for i, image_path in enumerate(images):
                        image = Image.open(image_path)
                        self._add_to_index(image, embed_func, doc_id, page_id=i + 1, metadata=metadata)
            elif item.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image = Image.open(item)
                self._add_to_index(image, embed_func, doc_id, metadata=metadata)
            else:
                raise ValueError(f"Unsupported input type: {item.suffix}")
        elif isinstance(item, Image.Image):
            self._add_to_index(item, embed_func, doc_id, metadata=metadata)
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")

    def _add_to_index(
        self,
        image: Image.Image,
        embed_func,
        doc_id: Union[str, int],
        page_id: int = 1,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        if any(entry["doc_id"] == doc_id and entry["page_id"] == page_id for entry in self.embed_id_to_doc_id.values()):
            raise ValueError(f"Document ID {doc_id} with page ID {page_id} already exists in the index")

        # Generate embedding
        embedding = embed_func(image)

        # Add to index
        embed_id = len(self.indexed_embeddings)
        self.indexed_embeddings.extend(list(torch.unbind(embedding)))
        self.embed_id_to_doc_id[embed_id] = {"doc_id": doc_id, "page_id": int(page_id)}

        # Update highest_doc_id
        self.highest_doc_id = max(self.highest_doc_id, int(doc_id) if isinstance(doc_id, int) else self.highest_doc_id)

        if self.full_document_collection:
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
                    print(f"Resizing image to {new_width}x{new_height}")
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

    def search(
        self,
        query: Union[str, List[str]],
        score_func,
        k: int = 10,
        return_base64_results: Optional[bool] = None,
    ) -> Union[List[Result], List[List[Result]]]:
        if return_base64_results is None:
            return_base64_results = bool(self.collection)

        k = min(k, len(self.indexed_embeddings))

        if isinstance(query, str):
            queries = [query]
        else:
            queries = query

        results = []
        for q in queries:
            scores = score_func(q, self.indexed_embeddings)
            top_pages = scores.argsort(axis=1)[0][-k:][::-1].tolist()

            query_results = []
            for embed_id in top_pages:
                doc_info = self.embed_id_to_doc_id[int(embed_id)]
                result = Result(
                    doc_id=doc_info["doc_id"],
                    page_num=int(doc_info["page_id"]),
                    score=float(scores[0][embed_id]),
                    metadata=self.doc_id_to_metadata.get(int(doc_info["doc_id"]), {}),
                    base64=(self.collection.get(int(embed_id)) if return_base64_results else None)
                )
                query_results.append(result)

            results.append(query_results)

        return results[0] if isinstance(query, str) else results

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
            "full_document_collection": self.full_document_collection,
            "highest_doc_id": self.highest_doc_id,
            "resize_stored_images": (True if self.max_image_width and self.max_image_height else False),
            "max_image_width": self.max_image_width,
            "max_image_height": self.max_image_height,
        }
        srsly.write_gzip_json(index_path / "index_config.json.gz", index_config)

        # Save embed_id_to_doc_id mapping
        srsly.write_gzip_json(index_path / "embed_id_to_doc_id.json.gz", self.embed_id_to_doc_id)

        # Save doc_ids_to_file_names
        srsly.write_gzip_json(index_path / "doc_ids_to_file_names.json.gz", self.doc_ids_to_file_names)

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

    def load_index(self, index_name: str):
        index_path = Path(self.index_root) / Path(index_name)
        if not index_path.exists():
            raise ValueError(f"Index {index_name} does not exist.")

        self.index_name = index_name
        index_config = srsly.read_gzip_json(index_path / "index_config.json.gz")
        self.full_document_collection = index_config.get("full_document_collection", False)
        self.max_image_width = index_config.get("max_image_width", None)
        self.max_image_height = index_config.get("max_image_height", None)

        # Load embeddings
        embeddings_path = index_path / "embeddings"
        embedding_files = sorted(embeddings_path.glob("embeddings_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
        self.indexed_embeddings = []
        for file in embedding_files:
            self.indexed_embeddings.extend(torch.load(file))

        # Load other data
        self.embed_id_to_doc_id = srsly.read_gzip_json(index_path / "embed_id_to_doc_id.json.gz")
        self.embed_id_to_doc_id = {int(k): v for k, v in self.embed_id_to_doc_id.items()}
        self.highest_doc_id = max(int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values())
        self.doc_ids = set(int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values())
        self.doc_ids_to_file_names = srsly.read_gzip_json(index_path / "doc_ids_to_file_names.json.gz")
        self.doc_ids_to_file_names = {int(k): v for k, v in self.doc_ids_to_file_names.items()}
        self.doc_id_to_metadata = srsly.read_gzip_json(index_path / "metadata.json.gz")
        self.doc_id_to_metadata = {int(k): v for k, v in self.doc_id_to_metadata.items()}

        # Load collection if using in-memory collection
        if self.full_document_collection:
            collection_path = index_path / "collection"
            json_files = sorted(collection_path.glob("*.json.gz"), key=lambda x: int(x.stem.split(".")[0]))
            for json_file in json_files:
                loaded_data = srsly.read_gzip_json(json_file)
                self.collection.update({int(k): v for k, v in loaded_data.items()})

        if self.verbose > 0:
            print(f"Index {index_name} loaded successfully.")

    def get_doc_ids_to_file_names(self):
        return self.doc_ids_to_file_names