import base64
import io
import logging
import os
import warnings
from pathlib import Path
import shutil
import srsly
import tempfile
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any, Callable

from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2_5, ColQwen2_5_Processor, ColQwen2Processor

# ColQwen3_5 added in colpali-engine 0.3.15
try:
    from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
    _COLQWEN3_5_AVAILABLE = True
except ImportError:
    _COLQWEN3_5_AVAILABLE = False

from .embedding_server import EmbeddingServerClient, EmbeddingServerConfig, EmbeddingServerManager
from pdf2image import convert_from_path
from PIL import Image
import torch

try:
    from .docling_ingest import chunk_pdf_to_images
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False

from .file_to_pdf import _convert_to_pdf
from .models_metadata import DocMetadata, MetadataFilter
from .objects import Result
from .plot_utils import draw_circle_on_max_patch, pil_from_base64, pil_to_base64_png, compute_patch_heatmap, majority_token_id, build_heatmap_overlays_base64
from .vector_store import (
    LocalVectorStore,
    MultiVectorQuery,
    SearchHit,
    StoredPoint,
    make_point_id,
    make_vector_store,
)

VERSION = "0.0.1"

# set the name for logging
logger = logging.getLogger(__name__)


# ColPaliModel supports three storage backends:
# - local backend: embeddings and mappings are stored in local files
# - qdrant backend: embeddings are stored in Qdrant (embedded), local sidecar files
#   still used for metadata, image caches, and heatmap-related tensors
# - milvus backend: embeddings stored in Milvus Lite (two-collection layout),
#   local sidecar files still used for metadata, image caches, and heatmap tensors
#
# All backends are accessed through the VectorStore interface defined in
# foretrieval/vector_store/.  This keeps ColPaliModel backend-agnostic and
# makes it easy to add a future remote-DB-server backend.
#
# The class is organized into:
# 1. initialization and loading
# 2. persistence
# 3. ingestion and indexing
# 4. search
# 5. result enrichment
# 6. file helpers

class ColPaliModel:
    # ============================================================
    # Initialization and index loading
    # ============================================================

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        index_name: Optional[str] = None,
        verbose: int = 1,
        load_from_index: bool = False,
        index_root: str = ".foretrieval",
        device: Optional[Union[str, torch.device]] = None,
        ingestion: Dict[str, Any] = {"backend": "default"},
        # New unified backend selection
        storage_backend: str = "local",
        storage_config: Optional[Dict[str, Any]] = None,
        # Deprecated: use storage_backend="qdrant" instead
        storage_qdrant: Optional[bool] = None,
        embedding_server: Optional[EmbeddingServerConfig] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "float16",
        **kwargs,
    ):
        if isinstance(pretrained_model_name_or_path, Path):
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        _supported = ("colpali", "colqwen2", "colqwen3")
        if not any(k in pretrained_model_name_or_path.lower() for k in _supported):
            raise ValueError(
                "FORetrieval supports ColPali, ColQwen2, ColQwen2.5, and ColQwen3.x models. "
                "Incorrect model name specified."
            )
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_name = self.pretrained_model_name_or_path
        self.verbose = verbose
        self.load_from_index = load_from_index
        self.index_root = index_root
        self.index_name = index_name
        self.kwargs = kwargs

        # Handle deprecated storage_qdrant boolean
        if storage_qdrant is not None:
            warnings.warn(
                "The 'storage_qdrant' parameter is deprecated and will be removed in a "
                "future release. Use storage_backend='qdrant' (or 'local') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            storage_backend = "qdrant" if storage_qdrant else "local"

        self.storage_backend = storage_backend.strip().lower()
        self.storage_config = storage_config or {}

        # Expose storage_qdrant as a read-only property for backward compatibility
        # (existing code that reads self.storage_qdrant still works)
        self._storage_qdrant_compat = (self.storage_backend == "qdrant")

        self.ingestion = ingestion
        self.ingestion_backend = (self.ingestion.get("backend") or "default").lower()
        if self.ingestion_backend == "docling":
            if not _DOCLING_AVAILABLE:
                raise RuntimeError(
                    "The 'docling' ingestion backend requires the docling package.\n"
                    "Install it with:  pip install \"foretrieval[docling]\"\n"
                    "or:               uv add foretrieval --extra docling"
                )
            self.docling_cfg = self.ingestion.get("docling_cfg", {})

        self.n_gpu = torch.cuda.device_count() if n_gpu == -1 else n_gpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(
                    "Used device is : %s", self.device 
                )
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype

        self.collection = {}
        self.embed_id_to_extra = {}
        self.doc_id_to_metadata = {}
        self.doc_ids_to_file_names = {}
        self.doc_ids = set()

        self.enable_heatmaps = False
        self.enable_circle = False
        self.SOURCE_EXTS = {".doc", ".docx", ".rtf", ".odt", ".ppt", ".pptx", ".odp", ".xls", ".xlsx", ".ods", ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".epub", ".html"}
        self.IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

        self.docling_dir = None
        if self.index_name is not None and self.ingestion_backend == "docling":
            self.docling_dir = Path(index_root) / self.index_name / "docling_chunks"
            self.docling_dir.mkdir(parents=True, exist_ok=True)

        # --- VectorStore ---
        self.vector_store = make_vector_store(self.storage_backend, self.storage_config)
        if self.index_name is not None:
            self.vector_store.open(
                self.index_name,
                Path(self.index_root),
                create=(not load_from_index),
            )
            if isinstance(self.vector_store, LocalVectorStore):
                self.vector_store.set_processor(None)  # set after model load

        # --- Remote embedding server ---
        self._remote_client: Optional[EmbeddingServerClient] = None
        if embedding_server is not None:
            if embedding_server.auto_deploy:
                manager = EmbeddingServerManager(embedding_server)
                manager.ensure_deployed()
            self._remote_client = EmbeddingServerClient(embedding_server)
            # In remote mode: load processor only (CPU), skip model weights.
            self._load_processor_only()
        else:
            self._load_model_and_processor()

        # Inject processor into local store now that it is loaded
        if isinstance(self.vector_store, LocalVectorStore) and hasattr(self, "processor"):
            self.vector_store.set_processor(self.processor)

        if not load_from_index:
            self.full_document_collection = False
            self.resize_stored_images = False
            self.max_image_width = None
            self.max_image_height = None
            self.highest_doc_id = -1
        else:
            self._load_index_state()

    # ------------------------------------------------------------------
    # Backward-compatibility property
    # ------------------------------------------------------------------

    @property
    def storage_qdrant(self) -> bool:
        """Deprecated compat property — use self.storage_backend instead."""
        return self._storage_qdrant_compat

    @storage_qdrant.setter
    def storage_qdrant(self, value: bool) -> None:
        self._storage_qdrant_compat = value

    def _resolve_model_and_processor_classes(self):
        """Return (model_cls, processor_cls) for the configured model name."""
        name = self.pretrained_model_name_or_path.lower()
        if "colpali" in name:
            return ColPali, ColPaliProcessor
        elif "colqwen3.5" in name or "colqwen3_5" in name:
            if not _COLQWEN3_5_AVAILABLE:
                raise ImportError(
                    "ColQwen3_5 requires colpali-engine>=0.3.15. "
                    "Upgrade with: pip install 'colpali-engine>=0.3.15'"
                )
            return ColQwen3_5, ColQwen3_5Processor
        elif "colqwen2.5" in name:
            return ColQwen2_5, ColQwen2_5_Processor
        else:
            # ColQwen2 and ColQwen3 (non-3.5) both use the ColQwen2 interface
            return ColQwen2, ColQwen2Processor

    def _load_model_and_processor(self):
        token = self.kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN")
        is_cuda = "cuda" in str(self.device) # ok for [Union[str, torch.device]]
        device_map = str(self.device) if is_cuda else None

        model_cls, processor_cls = self._resolve_model_and_processor_classes()

        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "4-bit/8-bit quantization requires the bitsandbytes package.\n"
                    "Install it with:  pip install \"foretrieval[quantization]\"\n"
                    "or:               uv add foretrieval --extra quantization"
                ) from exc
            if not is_cuda:
                raise ValueError(
                    "4-bit/8-bit quantization requires a CUDA device. "
                    f"Current device: {self.device}"
                )
            _dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            compute_dtype = _dtype_map.get(self.bnb_4bit_compute_dtype, torch.float16)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        load_kwargs: Dict[str, Any] = dict(
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token=token,
        )
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = model_cls.from_pretrained(
            self.pretrained_model_name_or_path,
            **load_kwargs,
        )
        self.processor = processor_cls.from_pretrained(
            self.pretrained_model_name_or_path,
            token=token,
        )

        self.model = self.model.eval()
        if device_map is None:
            self.model = self.model.to(self.device)

    def _load_processor_only(self):
        """Load only the processor (no model weights) for remote embedding mode."""
        token = self.kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN")
        _, processor_cls = self._resolve_model_and_processor_classes()
        self.processor = processor_cls.from_pretrained(
            self.pretrained_model_name_or_path,
            token=token,
        )
        self.model = None

    # ------------------------------------------------------------------
    # Bookkeeping persistence
    #
    # The index-level bookkeeping (model name, doc metadata, file-name map,
    # per-embedding extras and a few scalar flags) is normally written to
    # local sidecar files under ``index_root/index_name``.  When the active
    # vector store stores this on the server (``supports_remote_bookkeeping``),
    # we route the blob through the store instead so the client needs no local
    # index directory at all — this is the remote ``vector_db_server`` mode.
    # ------------------------------------------------------------------

    def _uses_remote_bookkeeping(self) -> bool:
        if self.storage_backend != "remote":
            return False
        supports = getattr(self.vector_store, "supports_remote_bookkeeping", None)
        try:
            return bool(supports()) if callable(supports) else False
        except Exception:
            return False

    def _build_bookkeeping_blob(self, description: str = "") -> Dict[str, Any]:
        """Assemble the in-memory bookkeeping into a single serialisable blob."""
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
            "storage_backend": self.storage_backend,
            # storage_config is intentionally omitted: in remote mode the
            # connection config comes from the caller (vector_db_server config),
            # never from persisted bookkeeping.
            "storage_config": None,
            "description": description,
        }
        return {
            "index_config": index_config,
            "embed_id_to_extra": self.embed_id_to_extra,
            "doc_ids_to_file_names": self.doc_ids_to_file_names,
            "doc_id_to_metadata": self.doc_id_to_metadata,
        }

    def _apply_bookkeeping_blob(self, blob: Dict[str, Any]) -> None:
        """Populate in-memory bookkeeping from a server-loaded blob."""
        index_config = blob.get("index_config", {}) or {}
        self.full_document_collection = index_config.get(
            "full_document_collection", False
        )
        self.resize_stored_images = index_config.get("resize_stored_images", False)
        self.max_image_width = index_config.get("max_image_width", None)
        self.max_image_height = index_config.get("max_image_height", None)
        self.index_description = index_config.get("description", "")

        self.embed_id_to_extra = {
            int(k): v for k, v in (blob.get("embed_id_to_extra") or {}).items()
        }
        self.doc_ids_to_file_names = {
            int(k): v for k, v in (blob.get("doc_ids_to_file_names") or {}).items()
        }
        self.doc_id_to_metadata = {
            int(k): v for k, v in (blob.get("doc_id_to_metadata") or {}).items()
        }
        # Derive doc_ids / highest_doc_id from the file-names map (always
        # present in bookkeeping, even when add_metadata=False).
        id_source = self.doc_ids_to_file_names or self.doc_id_to_metadata
        self.highest_doc_id = max(id_source.keys(), default=-1)
        self.doc_ids = set(id_source.keys())

    def _load_index_state(self):

        # Remote bookkeeping mode: pull everything from the server, no local
        # index directory is read.
        if self.storage_backend == "remote" and self._uses_remote_bookkeeping():
            blob = self.vector_store.load_bookkeeping()
            if blob is None:
                raise FileNotFoundError(
                    f"No bookkeeping found on the server for collection "
                    f"'{self.index_name}'. The collection may not have been "
                    "indexed yet."
                )
            self._apply_bookkeeping_blob(blob)
            return

        index_path = Path(self.index_root) / self.index_name
        index_config_path = index_path / "index_config.json.gz"
        index_config: dict = srsly.read_gzip_json(index_config_path)
        self.full_document_collection = index_config.get("full_document_collection", False)
        self.resize_stored_images = index_config.get("resize_stored_images", False)
        self.max_image_width = index_config.get("max_image_width", None)
        self.max_image_height = index_config.get("max_image_height", None)
        self.index_description = index_config.get("description", "")

        if self.full_document_collection:
            collection_path = index_path / "collection"
            json_files = sorted(
                collection_path.glob("*.json.gz"),
                key=lambda x: int(x.stem.split(".")[0]),
            )
            for json_file in json_files:
                loaded_data = srsly.read_gzip_json(json_file)
                self.collection.update({int(k): v for k, v in loaded_data.items()})

        # Load sidecar files (metadata, filenames, extras)
        self._load_local_sidecars(index_path)

        # Load vector store sidecar (embeddings for local; nothing for qdrant/milvus)
        self.vector_store.load_sidecar(index_path)

        # Inject metadata map and processor into local store
        if isinstance(self.vector_store, LocalVectorStore):
            self.vector_store.set_processor(self.processor)
            self.vector_store.set_doc_id_to_metadata(self.doc_id_to_metadata)

        # Derive doc_ids / highest_doc_id from the file-names map (always
        # written, even when add_metadata=False) so that a no-metadata index
        # loads correctly.  Fall back to the metadata map only when the
        # file-names map is empty (e.g. legacy in-memory-only indexes).
        id_source = self.doc_ids_to_file_names or self.doc_id_to_metadata
        self.highest_doc_id = max(id_source.keys(), default=-1)
        self.doc_ids = set(id_source.keys())

    def _load_local_sidecars(self, index_path: Path):
        extra_path = index_path / "embed_id_to_extra.pt"
        if extra_path.exists():
            self.embed_id_to_extra = torch.load(extra_path, map_location="cpu")
            self.embed_id_to_extra = {int(k): v for k, v in self.embed_id_to_extra.items()}
        else:
            self.embed_id_to_extra = {}

        doc_names_path = index_path / "doc_ids_to_file_names.json.gz"
        if doc_names_path.exists():
            self.doc_ids_to_file_names = srsly.read_gzip_json(doc_names_path)
            self.doc_ids_to_file_names = {int(k): v for k, v in self.doc_ids_to_file_names.items()}
        else:
            self.doc_ids_to_file_names = {}

        metadata_path = index_path / "metadata.json.gz"
        if metadata_path.exists():
            self.doc_id_to_metadata = srsly.read_gzip_json(metadata_path)
            self.doc_id_to_metadata = {int(k): v for k, v in self.doc_id_to_metadata.items()}
        else:
            self.doc_id_to_metadata = {}

    def set_enable_heatmaps_and_circle(self, enable_heatmaps: bool, enable_circle: bool):
        self.enable_heatmaps = enable_heatmaps
        self.enable_circle = enable_circle

    # ============================================================
    # Persistence and index export
    # ============================================================

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        ingestion: Dict[str, Any] = {"backend": "default"},
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = ".foretrieval",
        embedding_server: Optional[EmbeddingServerConfig] = None,
        storage_backend: str = "local",
        storage_config: Optional[Dict[str, Any]] = None,
        # Deprecated
        storage_qdrant: Optional[bool] = None,
        **kwargs,
    ):
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ingestion=ingestion,
            n_gpu=n_gpu,
            verbose=verbose,
            load_from_index=False,
            index_root=index_root,
            device=device,
            embedding_server=embedding_server,
            storage_backend=storage_backend,
            storage_config=storage_config,
            storage_qdrant=storage_qdrant,
            **kwargs,
        )

    @staticmethod
    def _fetch_remote_model_name(
        index_name: str, storage_config: Dict[str, Any]
    ) -> str:
        """Fetch the indexed model name from the remote server bookkeeping.

        Used by from_index() in remote mode to bootstrap the instance before
        any local state exists.  Raises if the server has no bookkeeping for
        the collection.
        """
        from .vector_store.factory import make_vector_store

        vs = make_vector_store("remote", storage_config)
        vs.open(index_name, Path("."), create=False)
        try:
            blob = vs.load_bookkeeping()
        finally:
            try:
                vs.close()
            except Exception:
                pass
        if not blob:
            raise FileNotFoundError(
                f"No bookkeeping found on the server for collection "
                f"'{index_name}'. Has it been indexed?"
            )
        model_name = (blob.get("index_config", {}) or {}).get("model_name")
        if not model_name:
            raise ValueError(
                f"Server bookkeeping for '{index_name}' has no model_name."
            )
        return model_name

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = ".foretrieval",
        embedding_server: Optional[EmbeddingServerConfig] = None,
        storage_backend: Optional[str] = None,
        storage_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Remote bookkeeping mode: the connection config is supplied by the
        # caller (from the vector_db_server config), and the model name plus
        # all index state live on the server — no local index directory is
        # read.  Triggered by storage_backend="remote".
        if (storage_backend or "").strip().lower() == "remote":
            index_name = Path(index_path).name
            model_name = cls._fetch_remote_model_name(index_name, storage_config or {})
            instance = cls(
                pretrained_model_name_or_path=model_name,
                n_gpu=n_gpu,
                index_name=index_name,
                verbose=verbose,
                load_from_index=True,
                index_root=index_root,
                device=device,
                storage_backend="remote",
                storage_config=storage_config,
                embedding_server=embedding_server,
                **kwargs,
            )
            return instance

        index_path = Path(os.path.join(Path(index_root), Path(index_path)))
        index_config: dict = srsly.read_gzip_json(index_path / "index_config.json.gz")
        disk_backend = index_config.get("storage_backend", "local")
        # Caller-supplied backend wins if given, else use on-disk value.
        storage_backend = (storage_backend or disk_backend)

        # For the remote backend, merge on-disk storage_config (URL, backend, …)
        # with caller-supplied storage_config (api_key, etc.).  Caller wins on
        # overlap so that credentials can be injected at load time without
        # modifying the on-disk index.
        disk_storage_config = index_config.get("storage_config") or {}
        if disk_storage_config and storage_backend == "remote":
            merged_storage_config = {**disk_storage_config, **(storage_config or {})}
        else:
            merged_storage_config = storage_config

        instance = cls(
            pretrained_model_name_or_path=index_config["model_name"],
            n_gpu=n_gpu,
            index_name=index_path.name,
            verbose=verbose,
            load_from_index=True,
            index_root=str(index_path.parent),
            device=device,
            storage_backend=storage_backend,
            storage_config=merged_storage_config,
            embedding_server=embedding_server,
            **kwargs,
        )
        instance.index_description = index_config.get("description", "")
        return instance

    def _export_index(self, description: str = ""):
        if self.index_name is None:
            raise ValueError("No index name specified. Cannot export.")

        # Remote bookkeeping mode: push everything to the server, no local
        # index directory is written.
        if self.storage_backend == "remote" and self._uses_remote_bookkeeping():
            if not description:
                existing = self.vector_store.load_bookkeeping()
                if existing:
                    description = (existing.get("index_config", {}) or {}).get(
                        "description", ""
                    )
            blob = self._build_bookkeeping_blob(description=description)
            self.vector_store.export_bookkeeping(blob)
            if self.verbose > 0:
                print(
                    f"Index bookkeeping stored on server for "
                    f"collection '{self.index_name}'"
                )
            return

        index_path = Path(self.index_root) / self.index_name
        index_path.mkdir(parents=True, exist_ok=True)

        # Preserve existing description on incremental updates when none is supplied
        if not description:
            cfg_path = index_path / "index_config.json.gz"
            if cfg_path.exists():
                try:
                    description = srsly.read_gzip_json(cfg_path).get("description", "")
                except Exception:
                    pass

        # For the remote backend, persist the storage_config so from_index()
        # can reconstruct the client.  Sensitive fields (api_key) are stripped
        # and must be re-supplied by the caller at load time.
        persisted_storage_config: Optional[Dict[str, Any]] = None
        if self.storage_backend == "remote" and self.storage_config:
            persisted_storage_config = {
                k: v
                for k, v in self.storage_config.items()
                if k != "api_key"
            }

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
            "storage_backend": self.storage_backend,
            "storage_config": persisted_storage_config,
            "description": description,
        }
        srsly.write_gzip_json(index_path / "index_config.json.gz", index_config)

        # Shared sidecar files
        torch.save(self.embed_id_to_extra, index_path / "embed_id_to_extra.pt")
        srsly.write_gzip_json(index_path / "doc_ids_to_file_names.json.gz", self.doc_ids_to_file_names)
        srsly.write_gzip_json(index_path / "metadata.json.gz", self.doc_id_to_metadata)

        if self.full_document_collection:
            collection_path = index_path / "collection"
            collection_path.mkdir(exist_ok=True)
            for i in range(0, len(self.collection), 500):
                chunk = dict(list(self.collection.items())[i : i + 500])
                srsly.write_gzip_json(collection_path / f"{i}.json.gz", chunk)

        # Delegate vector persistence to the backend
        self.vector_store.export_sidecar(index_path)

        if self.verbose > 0:
            print(f"Index exported to {index_path}")

    # ============================================================
    # Index building and ingestion
    # ============================================================

    def _cleanup_failed_index(self, index_name: str) -> None:
        """Remove artefacts created by a failed ``index()`` call.

        Called when indexing raises an exception after the vector store and/or
        local index directory were already created, so the caller can retry
        with the same index name without hitting "index already exists".

        For the local backend the index directory is removed with
        ``shutil.rmtree``.  For the remote (server-side bookkeeping) backend
        the remote collection is deleted via the server client.  Errors during
        cleanup are logged as warnings and swallowed so that the original
        exception propagates cleanly.
        """
        # Reset in-memory state so the object is reusable after cleanup.
        self.index_name = None
        self.highest_doc_id = -1
        self.doc_ids = set()
        self.doc_ids_to_file_names = {}
        self.doc_id_to_metadata = {}

        # Local backend: remove the partial index directory.
        index_path = Path(self.index_root) / index_name
        if index_path.exists():
            try:
                shutil.rmtree(index_path)
                logger.info(
                    "Cleaned up partial local index directory: %s", index_path
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not remove partial index directory %s: %s",
                    index_path,
                    exc,
                )

        # Remote backend: delete the collection on the server.
        if self.storage_backend == "remote" and self._uses_remote_bookkeeping():
            from .vector_store.remote import RemoteVectorStore
            if isinstance(self.vector_store, RemoteVectorStore):
                try:
                    self.vector_store._client.delete_collection(index_name)
                    logger.info(
                        "Cleaned up partial remote collection: %s", index_name
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not delete remote collection %s: %s",
                        index_name,
                        exc,
                    )

    def index(
        self,
        input_path: Union[str, Path],
        index_name: Optional[str] = None,
        doc_ids: Optional[List[int]] = None,
        store_collection_with_index: bool = False,
        overwrite: bool = False,
        metadata: Optional[Union[List[DocMetadata], Dict[int, DocMetadata]]] = None,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
        batch_size: int = 1,
        description: str = "",
        ai_cfg: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Union[Dict[int, str], None]:
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
        if index_name is None:
            raise ValueError("index_name must be specified to create a new index.")

        index_path = Path(os.path.join(Path(self.index_root), Path(index_name)))
        # In remote bookkeeping mode the local index directory is irrelevant
        # (vectors and bookkeeping live on the server), so skip the local-dir
        # existence guard entirely.
        if (
            not (self.storage_backend == "remote" and self._uses_remote_bookkeeping())
            and index_path.exists()
        ):
            if not overwrite and (
                (index_path.is_dir() and len(list(index_path.iterdir())) > 0)
                or index_path.is_file()
            ):
                logger.warning(
                    f"An index named {index_name} already exists.",
                    "Use overwrite=True to delete the existing index and build a new one.",
                    "Exiting indexing without doing anything...",
                )
                return None
            else:
                logger.info(
                    f"overwrite is on. Deleting existing index {index_name} to build a new one."
                )
                shutil.rmtree(index_path)

        if store_collection_with_index:
            self.full_document_collection = True
        self.index_name = index_name

        # Open / create vector store for the new index
        self.vector_store.open(
            index_name,
            Path(self.index_root),
            create=True,
        )
        if isinstance(self.vector_store, LocalVectorStore):
            self.vector_store.set_processor(self.processor)
            self.vector_store.set_doc_id_to_metadata(self.doc_id_to_metadata)

        self.max_image_width = max_image_width
        self.max_image_height = max_image_height

        input_path = Path(input_path)
        if not hasattr(self, "highest_doc_id") or overwrite is True:
            self.highest_doc_id = -1

        # Validate before entering the try block so that validation errors also
        # trigger cleanup of the just-opened vector store / local directory.
        if input_path.is_dir():
            items = sorted(
                (p for p in input_path.rglob("*") if p.is_file()),
                key=lambda p: p.relative_to(input_path),
            )
            if doc_ids is not None and len(doc_ids) != len(items):
                self._cleanup_failed_index(index_name)
                raise ValueError(
                    f"Number of doc_ids ({len(doc_ids)}) does not match number of documents ({len(items)})"
                )
            if metadata is not None and len(metadata) != len(items):
                self._cleanup_failed_index(index_name)
                raise ValueError(
                    f"Number of metadata entries ({len(metadata)}) does not match number of documents ({len(items)})"
                )
        else:
            items = None  # single-file path handled below

        try:
            if items is not None:
                # Directory indexing path
                n_files = len(items)
                if on_progress is not None:
                    try:
                        on_progress({"stage": "start", "n_files": n_files})
                    except Exception:  # noqa: BLE001
                        pass
                for i, item in tqdm(
                    enumerate(items), total=n_files, desc="Indexing files"
                ):
                    doc_id = doc_ids[i] if doc_ids else self.highest_doc_id + 1
                    if metadata is None:
                        doc_md = None
                    elif isinstance(metadata, list):
                        doc_md = metadata[i]
                    elif isinstance(metadata, dict):
                        doc_md = metadata.get(doc_id)
                    else:
                        doc_md = metadata[doc_id] if metadata else None

                    if on_progress is not None:
                        try:
                            on_progress({
                                "stage": "file_start",
                                "file": item.name,
                                "file_idx": i,
                                "n_files": n_files,
                            })
                        except Exception:  # noqa: BLE001
                            pass

                    try:
                        self.add_to_index(
                            item,
                            store_collection_with_index,
                            doc_id=doc_id,
                            metadata=doc_md,
                            batch_size=batch_size,
                            on_progress=on_progress,
                            _file_idx=i,
                            _n_files=n_files,
                        )
                    except Exception as e:
                        logger.warning(f"Skipping faulty PDF {item}:\n{str(e)}")
                        continue

                    if on_progress is not None:
                        try:
                            on_progress({
                                "stage": "file_done",
                                "file": item.name,
                                "file_idx": i,
                                "n_files": n_files,
                            })
                        except Exception:  # noqa: BLE001
                            pass

            else:
                # Single-file indexing path
                if metadata is not None and len(metadata) != 1:
                    raise ValueError(
                        "For a single document, metadata should be a list with one dictionary"
                    )
                doc_id = doc_ids[0] if doc_ids else self.highest_doc_id + 1
                doc_metadata = metadata[0] if metadata else None
                if on_progress is not None:
                    try:
                        on_progress({"stage": "start", "n_files": 1})
                        on_progress({
                            "stage": "file_start",
                            "file": input_path.name,
                            "file_idx": 0,
                            "n_files": 1,
                        })
                    except Exception:  # noqa: BLE001
                        pass
                self.add_to_index(
                    input_path,
                    store_collection_with_index,
                    doc_id=doc_id,
                    metadata=doc_metadata,
                    on_progress=on_progress,
                    _file_idx=0,
                    _n_files=1,
                )
                if on_progress is not None:
                    try:
                        on_progress({
                            "stage": "file_done",
                            "file": input_path.name,
                            "file_idx": 0,
                            "n_files": 1,
                        })
                    except Exception:  # noqa: BLE001
                        pass

            if on_progress is not None:
                try:
                    on_progress({"stage": "all_done"})
                except Exception:  # noqa: BLE001
                    pass

            # Auto-generate index description from per-doc AI metadata when available
            if not description and ai_cfg and self.doc_id_to_metadata:
                from .metadata import generate_index_description
                description = generate_index_description(self.doc_id_to_metadata, ai_cfg)

            self._export_index(description=description)
            if self.highest_doc_id == -1:
                logger.warning("No documents were indexed.")

        except Exception:
            # Indexing failed after the vector store / local directory were
            # already created.  Remove the partial artefacts so the caller can
            # retry cleanly.
            self._cleanup_failed_index(index_name)
            raise

        return self.doc_ids_to_file_names

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        store_collection_with_index: bool,
        doc_id: Optional[Union[int, List[int]]] = None,
        metadata: Optional[Union[List[DocMetadata], DocMetadata]] = None,
        batch_size: int = 1,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        _file_idx: int = 0,
        _n_files: int = 1,
    ) -> Dict[int, str]:
        if self.index_name is None:
            raise ValueError(
                "No index loaded. Use index() to create or load an index first."
            )
        if not hasattr(self, "highest_doc_id"):
            self.highest_doc_id = -1

        # Ensure vector store is open for this index.
        # We check if the client is initialised (not whether the collection exists),
        # to avoid re-opening when the collection was just opened but not yet populated.
        if not self._vector_store_is_open():
            self.vector_store.open(
                self.index_name,
                Path(self.index_root),
                create=True,
            )
            if isinstance(self.vector_store, LocalVectorStore):
                self.vector_store.set_processor(self.processor)
                self.vector_store.set_doc_id_to_metadata(self.doc_id_to_metadata)

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

        if doc_ids and len(doc_ids) != len(input_items):
            raise ValueError(
                f"Number of doc_ids ({len(doc_ids)}) does not match number of input items ({len(input_items)})"
            )

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
                        batch_size,
                        on_progress=on_progress,
                        _file_idx=_file_idx,
                        _n_files=_n_files,
                    )
                else:
                    stored_path = self._process_and_add_to_index(
                        item_path,
                        store_collection_with_index,
                        current_doc_id,
                        current_metadata,
                        batch_size,
                        on_progress=on_progress,
                        _file_idx=_file_idx,
                        _n_files=_n_files,
                    )
                    if stored_path is None:
                        self.doc_ids_to_file_names[current_doc_id] = "In-memory Image"
                    else:
                        self.doc_ids_to_file_names[current_doc_id] = str(stored_path)

            elif isinstance(item, Image.Image):
                self._process_and_add_to_index(
                    item, store_collection_with_index, current_doc_id, current_metadata,
                    on_progress=on_progress,
                    _file_idx=_file_idx,
                    _n_files=_n_files,
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
        batch_size: int,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        _file_idx: int = 0,
        _n_files: int = 1,
    ):
        files = sorted(
            (p for p in directory.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(directory),
        )
        for i, item in tqdm(enumerate(files), total=len(files), desc=f"Indexing {directory.name}"):
            logger.debug(f"Indexing file: {item}")
            current_doc_id = base_doc_id + i
            stored_path = self._process_and_add_to_index(
                item, store_collection_with_index, current_doc_id, metadata, batch_size,
                on_progress=on_progress,
                _file_idx=_file_idx,
                _n_files=_n_files,
            )
            if stored_path is None:
                self.doc_ids_to_file_names[current_doc_id] = "In-memory Image"
            else:
                self.doc_ids_to_file_names[current_doc_id] = str(stored_path)

    def _process_and_add_to_index(
        self,
        item: Union[Path, Image.Image],
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        metadata: Optional[Dict[str, Union[str, int]]] = None,
        batch_size: int = 1,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        _file_idx: int = 0,
        _n_files: int = 1,
    ) -> Optional[Path]:
        """
        Process and index an image or any file (converted to PDF if needed).
        Returns the 'canonical' path (PDF or image) used, or None for in-memory images.
        """
        def _emit_page(file_name: str, page_idx: int, n_pages: int) -> None:
            if on_progress is None:
                return
            try:
                on_progress({
                    "stage": "page",
                    "file": file_name,
                    "file_idx": _file_idx,
                    "n_files": _n_files,
                    "page_idx": page_idx,
                    "n_pages": n_pages,
                })
            except Exception:  # noqa: BLE001
                pass

        if isinstance(item, Path):
            ext = item.suffix.lower()

            # 0) docling chunking (if enabled)
            if self.ingestion_backend == "docling":

                if ext == ".pdf":
                    pdf_file = item.resolve()
                else:
                    existing_pdf = self._find_existing_pdf(item)
                    if existing_pdf is not None:
                        pdf_file = existing_pdf
                    else:
                        pdf_file = _convert_to_pdf(item)
                        if pdf_file is None:
                            logger.warning(f"Docling ingestion: failed to convert {item} to PDF. Skipping.")
                            return None

                if self.docling_dir is None:
                    assert self.index_name is not None, "index_name must be set to use docling ingestion"
                    self.docling_dir = Path(self.index_root) / self.index_name / "docling_chunks"
                    self.docling_dir.mkdir(parents=True, exist_ok=True)
                chunks = chunk_pdf_to_images(pdf_file, output_dir=self.docling_dir)

                n_chunks = len(chunks)
                for i in range(0, n_chunks, batch_size):
                    batch_chunks, batch_page_ids, batch_chunk_ids = [], [], []
                    for j in range(i, min(i + batch_size, n_chunks)):
                        ch = chunks[j]
                        image = Image.open(ch.path)
                        batch_chunks.append(image)
                        batch_page_ids.append(ch.page_id)
                        batch_chunk_ids.append(ch.elem_id)
                        _emit_page(item.name, j, n_chunks)
                    self._add_to_index(
                        batch_chunks,
                        store_collection_with_index,
                        doc_id,
                        page_ids=batch_page_ids,
                        chunk_ids=batch_chunk_ids,
                        metadata=metadata,
                    )

                return Path(pdf_file).resolve()

            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]:
                image = Image.open(item)
                _emit_page(item.name, 0, 1)
                self._add_to_index(image, store_collection_with_index, doc_id, metadata=metadata)
                return item.resolve()

            elif ext == ".pdf":
                pdf_file = item.resolve()

                with tempfile.TemporaryDirectory() as path:
                    images = convert_from_path(
                        pdf_file,
                        thread_count=os.cpu_count() - 1,
                        output_folder=path,
                        paths_only=True,
                    )
                    n_pages = len(images)
                    for i in range(0, n_pages, batch_size):
                        batch_images, batch_page_ids = [], []
                        for j in range(i, min(i + batch_size, n_pages)):
                            image_path = images[j]
                            image = Image.open(image_path)
                            batch_images.append(image)
                            batch_page_ids.append(j + 1)
                            _emit_page(item.name, j, n_pages)
                        self._add_to_index(
                            batch_images,
                            store_collection_with_index,
                            doc_id,
                            page_ids=batch_page_ids,
                            metadata=metadata,
                        )
                return pdf_file
            else:
                existing_pdf = self._find_existing_pdf(item)
                if existing_pdf is not None:
                    pdf_file = existing_pdf
                else:
                    pdf_file = _convert_to_pdf(item)
                    if pdf_file is None:
                        return None

                with tempfile.TemporaryDirectory() as path:
                    images = convert_from_path(
                        pdf_file,
                        thread_count=os.cpu_count() - 1,
                        output_folder=path,
                        paths_only=True,
                    )
                    n_pages = len(images)
                    for i in range(0, n_pages, batch_size):
                        batch_images, batch_page_ids = [], []
                        for j in range(i, min(i + batch_size, n_pages)):
                            image_path = images[j]
                            image = Image.open(image_path)
                            batch_images.append(image)
                            batch_page_ids.append(j + 1)
                            _emit_page(item.name, j, n_pages)
                        self._add_to_index(
                            batch_images,
                            store_collection_with_index,
                            doc_id,
                            page_ids=batch_page_ids,
                            metadata=metadata,
                        )
                return Path(pdf_file).resolve()

        elif isinstance(item, Image.Image):
            _emit_page("<in-memory>", 0, 1)
            self._add_to_index(item, store_collection_with_index, doc_id, metadata=metadata)
            return None
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")

    def _add_to_index(
        self,
        images: Union[Image.Image, List[Image.Image]],
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        page_ids: Union[int, List[int]] = 1,
        chunk_ids: Optional[Union[int, List[int]]] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        # Convert single image to list for uniform processing
        if isinstance(images, Image.Image):
            images = [images]

        if isinstance(page_ids, int):
            page_ids = [page_ids]

        if chunk_ids is None:
            chunk_ids = [None] * len(images)
        elif isinstance(chunk_ids, int):
            chunk_ids = [chunk_ids]

        if len(images) != len(page_ids):
            raise ValueError(f"Number of images ({len(images)}) does not match number of page_ids ({len(page_ids)})")
        if len(images) != len(chunk_ids):
            raise ValueError(f"Number of images ({len(images)}) does not match number of chunk_ids ({len(chunk_ids)})")

        # Check for existing entries
        for page_id, chunk_id in zip(page_ids, chunk_ids):
            pid = make_point_id(int(doc_id), int(page_id), int(chunk_id) if chunk_id is not None else None)
            if self.vector_store.point_exists(pid):
                if chunk_id is not None:
                    raise ValueError(f"Document ID {doc_id} with chunk ID {chunk_id} already exists in the index")
                raise ValueError(f"Document ID {doc_id} with page ID {page_id} already exists in the index")

        # Process images locally (CPU) for heatmap sidecars
        processed_images = self.processor.process_images(images)
        input_ids_cpu = processed_images["input_ids"].detach().cpu()
        grid_cpu = processed_images.get("image_grid_thw")
        if grid_cpu is not None:
            grid_cpu = grid_cpu.detach().cpu()
        orig_sizes = [img.size for img in images]

        # Generate embeddings — remote or local path
        if self._remote_client is not None:
            embeddings_list = self._remote_client.embed_images(images)
        else:
            with torch.inference_mode():
                processed_images_gpu = {
                    k: v.to(self.device).to(
                        self.model.dtype
                        if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                        else v.dtype
                    )
                    for k, v in processed_images.items()
                }
                embeddings = self.model(**processed_images_gpu)
            embeddings_list = list(torch.unbind(embeddings.to("cpu")))

        # Determine dim for lazy collection creation
        dim = int(embeddings_list[0].shape[-1])
        if not self.vector_store.collection_exists():
            self.vector_store.create_collection(dim)

        # Store metadata
        if metadata is not None:
            md_jsonable = (
                metadata.as_jsonable() if isinstance(metadata, DocMetadata)
                else (DocMetadata(**metadata).as_jsonable() if isinstance(metadata, dict) else metadata)
            )
            self.doc_id_to_metadata[int(doc_id)] = md_jsonable
            if isinstance(self.vector_store, LocalVectorStore):
                self.vector_store.set_doc_id_to_metadata(self.doc_id_to_metadata)

        # Build StoredPoints and upsert
        points_to_upsert = []
        for i, (embedding, page_id, chunk_id) in enumerate(zip(embeddings_list, page_ids, chunk_ids)):
            pid = make_point_id(int(doc_id), int(page_id), int(chunk_id) if chunk_id is not None else None)

            payload = {
                "doc_id": int(doc_id),
                "page_id": int(page_id),
                "chunk_id": int(chunk_id) if chunk_id is not None else None,
                "metadata": (
                    metadata.as_jsonable() if isinstance(metadata, DocMetadata)
                    else (DocMetadata(**metadata).as_jsonable() if isinstance(metadata, dict) else {})
                ) if metadata is not None else {},
            }

            points_to_upsert.append(StoredPoint(
                point_id=pid,
                vector=embedding.cpu(),
                payload=payload,
            ))

            # Heatmap sidecar
            self.embed_id_to_extra[pid] = {
                "input_ids": input_ids_cpu[i],
                "image_grid_thw": grid_cpu[i] if grid_cpu is not None else None,
                "orig_size": orig_sizes[i],
            }

            if store_collection_with_index:
                img_str = self._post_process_image(images[i])
                self.collection[int(pid)] = img_str

        self.vector_store.upsert(points_to_upsert)
        self.doc_ids.add(int(doc_id))

    # ============================================================
    # Index maintenance
    # ============================================================

    def update_index_from_folder(
        self,
        folder: Union[str, Path],
        store_collection_with_index: bool = False,
        metadata_provider: Optional[Callable] = None,
        batch_size: int = 1,
        reindex_modified: bool = False,
    ) -> Dict[int, str]:
        """
        Adds only NEW files from a folder to the current index.
        """
        folder = Path(folder)
        assert folder.is_dir(), f"{folder} n'est pas un dossier existant."

        known = self._already_indexed_paths()

        inverse_map: Dict[str, int] = {}
        for did, p in self.doc_ids_to_file_names.items():
            if p and p != "In-memory Image":
                try:
                    inverse_map[str(Path(p).resolve())] = int(did)
                except Exception:
                    inverse_map[p] = int(did)

        added = 0
        updated = 0

        for item in sorted(
            (p for p in folder.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(folder),
        ):

            ext = item.suffix.lower()

            if ext == ".pdf" and self._is_mirror_pdf(item):
                if self.verbose > 1:
                    print(f"[skip] Mirror PDF ignored: {item}")
                continue

            cand_keys = self._candidate_keys(item)
            if any(k in known for k in cand_keys) and not reindex_modified:
                if self.verbose > 1:
                    print(f"[skip] Already indexed: {item}")
                continue

            if reindex_modified:
                target_key = None
                for k in cand_keys:
                    if k in known:
                        target_key = k
                        break

                if target_key is not None:
                    try:
                        src_stat = item.stat().st_mtime
                        tgt_stat = Path(target_key).stat().st_mtime
                    except Exception:
                        src_stat, tgt_stat = None, None

                    if (
                        src_stat is not None
                        and tgt_stat is not None
                        and src_stat <= tgt_stat
                    ):
                        if self.verbose > 1:
                            print(f"[skip] Unchanged (mtime): {item}")
                        continue

                    old_doc_id = inverse_map.get(target_key)
                    if old_doc_id is not None:
                        if self.verbose > 0:
                            print(
                                f"[update] Reindexing (modified): {item} (doc_id {old_doc_id})"
                            )
                        updated += 1

            doc_id = self.highest_doc_id + 1
            md = metadata_provider(item) if metadata_provider else None
            stored_path = self._process_and_add_to_index(
                item,
                store_collection_with_index=store_collection_with_index,
                doc_id=doc_id,
                metadata=md,
                batch_size=batch_size,
            )
            if stored_path is None:
                self.doc_ids_to_file_names[doc_id] = "In-memory Image"
            else:
                self.doc_ids_to_file_names[doc_id] = str(Path(stored_path).resolve())

            self.doc_ids.add(doc_id)
            self.highest_doc_id = max(self.highest_doc_id, doc_id)
            added += 1

        self._export_index()

        if self.verbose > 0:
            print(f"[incr] added: {added} | reindexed: {updated}")

        return self.doc_ids_to_file_names

    def remove_from_index(self):
        raise NotImplementedError("This method is not implemented yet.")

    # ============================================================
    # Search
    # ============================================================

    def _encode_search_query(self, query: str):
        if self._remote_client is not None:
            return self._remote_client.embed_query(query)

        with torch.inference_mode():
            batch_query = self.processor.process_queries([query])
            batch_query = {
                kk: vv.to(self.device).to(
                    self.model.dtype
                    if vv.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else vv.dtype
                )
                for kk, vv in batch_query.items()
            }
            embeddings_query = self.model(**batch_query)
            qs = list(torch.unbind(embeddings_query.to("cpu")))

        input_ids = batch_query["input_ids"][0].detach().cpu().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids)
        valid_idxs = [i for i, tok in enumerate(tokens) if tok not in {"<|endoftext|>", "Query", ":"}]
        return [qs[0][valid_idxs]]

    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, str]] = None,
        return_base64_results: Optional[bool] = None
    ) -> List[Result]:

        if return_base64_results is None:
            return_base64_results = bool(self.collection)

        if k < 1:
            return []

        qs = self._encode_search_query(query)

        mvq = MultiVectorQuery(
            vectors=qs[0],
            filter_metadata=filter_metadata,
        )
        hits = self.vector_store.search(mvq, k)

        results = self._hits_to_results(hits, qs[0], k, return_base64_results)
        return self._finalize_results(results, return_base64_results)

    def _hits_to_results(
        self,
        hits: List[SearchHit],
        q_emb: torch.Tensor,
        k: int,
        return_base64_results: bool,
    ) -> List[Result]:
        results: List[Result] = []
        for hit in hits:
            payload = hit.payload
            doc_id = int(payload.get("doc_id", 0))
            page_id = int(payload.get("page_id", 1))
            chunk_id = payload.get("chunk_id")

            result = Result(
                doc_id=doc_id,
                page_num=page_id,
                chunk_num=int(chunk_id) if chunk_id is not None else None,
                score=float(hit.score), #force float conversion to avoid pyTorch Tensors
                metadata=payload.get("metadata", self.doc_id_to_metadata.get(doc_id, {})),
                base64=self.collection.get(hit.point_id) if return_base64_results else None,
            )

            extra = self.embed_id_to_extra.get(hit.point_id)
            if (self.enable_heatmaps or self.enable_circle) and extra is not None:
                p_emb = self.vector_store.fetch_vector(hit.point_id)
                if p_emb is not None:
                    result = self._attach_heatmaps_local(
                        result=result,
                        q_emb=q_emb,
                        p_emb=p_emb,
                        extra=extra,
                        k=k,
                    )

            results.append(result)

        return results

    def filter_embeddings(self, filter_metadata: Union[Dict[str, Any], MetadataFilter]):
        """Legacy method kept for backward compat. Use search(filter_metadata=...) instead."""
        if not isinstance(self.vector_store, LocalVectorStore):
            raise NotImplementedError(
                "filter_embeddings() is only supported for the local backend. "
                "Use search(filter_metadata=...) for other backends."
            )
        f = (
            filter_metadata
            if isinstance(filter_metadata, MetadataFilter)
            else MetadataFilter(**filter_metadata)
        )
        return self.vector_store._filter_by_metadata(f)

    # ============================================================
    # Result enrichment and visualization
    # ============================================================

    def _get_image_token_id_from_extra(self, extra: dict) -> int:
        if hasattr(self.processor, "image_token_id"):
            return int(self.processor.image_token_id)
        return majority_token_id(extra["input_ids"])

    def _attach_heatmaps_local(self, result: Result, q_emb, p_emb, extra: dict, k: int) -> Result:
        img_tok = self._get_image_token_id_from_extra(extra)

        result.metadata = dict(result.metadata or {})
        heat_soft, heat_global = None, None

        if self.enable_circle or self.enable_heatmaps:
            heat_soft, _ = compute_patch_heatmap(
                q_emb=q_emb,
                p_emb=p_emb,
                input_ids=extra["input_ids"],
                image_grid_thw=extra["image_grid_thw"],
                image_token_id=img_tok,
                mode="soft_topk",
                topk=min(k, 8),
                temperature=0.2,
                normalize=False,
            )

        if self.enable_heatmaps:
            heat_global, _ = compute_patch_heatmap(
                q_emb=q_emb,
                p_emb=p_emb,
                input_ids=extra["input_ids"],
                image_grid_thw=extra["image_grid_thw"],
                image_token_id=img_tok,
                mode="global_sum",
                topk=k,
                temperature=0.2,
                normalize=False,
            )

        hm = {"soft_topk": {"heat_2d": heat_soft}}
        if self.enable_heatmaps:
            hm["global_sum"] = {"heat_2d": heat_global}
        result.metadata["heatmaps"] = hm

        return result

    def _finalize_results(self, results: List[Result], return_base64_results: bool) -> List[Result]:
        if not return_base64_results:
            return results

        for r in results:
            self.fetch_result_img(r)

        for r in results:
            if not r.base64:
                continue

            meta = r.metadata or {}
            need_overlay, need_circle = bool(self.enable_heatmaps), bool(self.enable_circle)

            if not (need_overlay or need_circle):
                continue

            hm = meta.get("heatmaps") or {}
            img = None
            if (need_overlay and hm) or need_circle:
                img = pil_from_base64(r.base64)

            if need_overlay and hm:
                meta["heatmap_overlays_base64"] = build_heatmap_overlays_base64(
                    img=img,
                    heatmaps=hm,
                    interps=("nearest", "bilinear"),
                    alpha=0.45,
                    cmap="jet",
                    shift_x=0.0,
                    shift_y=0.0,
                    patch_grow_pct=300.0,
                    grow_mode="mean",
                )

            if need_circle:
                soft = (hm.get("soft_topk") or {}).get("heat_2d")
                if soft is not None:
                    img_marked = draw_circle_on_max_patch(
                        img=img,
                        heat_2d=soft,
                        patch_grow_pct=300.0,
                        grow_mode="mean",
                    )
                    meta["soft_topk_max_patch_circle_base64"] = pil_to_base64_png(img_marked)

            r.metadata = meta

        return results

    def fetch_result_img(self, result: Result) -> Result:
        if result.base64:
            return result

        doc_id = result.doc_id
        file_name = self.doc_ids_to_file_names.get(doc_id)
        if not file_name or file_name == "In-memory Image":
            return result

        path = Path(file_name)

        if self.ingestion_backend == "docling":
            try:
                if self.docling_dir is None:
                    assert self.index_name is not None, "index_name must be set to use docling ingestion"
                    self.docling_dir = Path(self.index_root) / self.index_name / "docling_chunks"
                    self.docling_dir.mkdir(parents=True, exist_ok=True)
                assert result.chunk_num is not None, "Result.chunk_num must be defined"
                path_chunk = Path(self.docling_dir) / f"{path.stem}_p{result.page_num}_{result.chunk_num}.png"
                assert path_chunk.exists(), f"Path {path_chunk} for chunk {result.chunk_num} does not exists"
                image = Image.open(path_chunk)
                result.base64 = self._post_process_image(image)
                return result
            except Exception as e:
                if self.verbose > 0:
                    logger.warning(f"[fetch_result_img] Docling chunk fetch error: {e}")

        ext = path.suffix.lower()

        try:
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]:
                image = Image.open(path)
                result.base64 = self._post_process_image(image)
                return result

            if ext != ".pdf":
                sibling_pdf = self._find_existing_pdf(path)
                if sibling_pdf is not None:
                    self.doc_ids_to_file_names[doc_id] = str(sibling_pdf)
                    path = sibling_pdf
                    ext = ".pdf"
                else:
                    pdf_path = _convert_to_pdf(path)
                    if pdf_path and pdf_path.exists():
                        self.doc_ids_to_file_names[doc_id] = str(pdf_path)
                        path = pdf_path
                        ext = ".pdf"
                    else:
                        if self.verbose > 0:
                            print(
                                f"[fetch_result_img] Impossible de convertir {path} en PDF."
                            )
                        return result

            with tempfile.TemporaryDirectory() as tmpdir:
                images = convert_from_path(
                    str(path),
                    thread_count=os.cpu_count() - 1,
                    first_page=result.page_num,
                    last_page=result.page_num,
                    paths_only=True,
                    output_folder=tmpdir,
                )
                image = Image.open(images[0])
                result.base64 = self._post_process_image(image)
            return result

        except Exception as e:
            if self.verbose > 0:
                print(f"[fetch_result_img] Erreur: {e}")
            return result

    def _post_process_image(self, image: Image.Image) -> str:
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
                    f"compression {new_width / img_width * new_height / img_height:.2f})",
                )
            image = image.resize((new_width, new_height), Image.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    # ============================================================
    # File helpers
    # ============================================================

    def _looks_like_pdf(self, path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size < 5:
                return False
            with open(path, "rb") as f:
                return f.read(5) == b"%PDF-"
        except Exception:
            return False

    def _find_existing_pdf(self, src: Path) -> Optional[Path]:
        cand = src.with_suffix(".pdf")
        if cand.exists() and self._looks_like_pdf(cand):
            return cand.resolve()
        return None

    def _already_indexed_paths(self) -> set:
        vals = set()
        for p in self.doc_ids_to_file_names.values():
            if not p or p == "In-memory Image":
                continue
            try:
                vals.add(str(Path(p).resolve()))
            except Exception:
                vals.add(p)
        return vals

    def _candidate_keys(self, path: Path) -> List[str]:
        keys = []
        try:
            keys.append(str(path.resolve()))
        except Exception:
            keys.append(str(path))

        sibling_pdf = path.with_suffix(".pdf")
        if sibling_pdf.exists() and self._looks_like_pdf(sibling_pdf):
            try:
                keys.append(str(sibling_pdf.resolve()))
            except Exception:
                keys.append(str(sibling_pdf))
        return keys

    def _is_mirror_pdf(self, path: Path) -> bool:
        if path.suffix.lower() != ".pdf":
            return False
        stem = path.with_suffix("")
        parent = path.parent
        for ext in self.SOURCE_EXTS:
            if Path(os.path.join(parent, f"{stem.name}{ext}")).exists():
                return True
        return False

    def _vector_store_is_open(self) -> bool:
        """Return True if the vector store has an active client connection."""
        from .vector_store.qdrant import QdrantVectorStore
        from .vector_store.milvus import MilvusVectorStore
        from .vector_store.remote import RemoteVectorStore
        if isinstance(self.vector_store, LocalVectorStore):
            return self.vector_store._index_name is not None
        if isinstance(self.vector_store, QdrantVectorStore):
            return self.vector_store._client is not None
        if isinstance(self.vector_store, MilvusVectorStore):
            return self.vector_store._client is not None
        if isinstance(self.vector_store, RemoteVectorStore):
            return self.vector_store.is_opened
        return False

    # ============================================================
    # Accessors for backward compat
    # ============================================================

    def get_doc_ids_to_file_names(self):
        return self.doc_ids_to_file_names

    @property
    def indexed_embeddings(self):
        """Backward-compat: return embedding list for local backend only."""
        if isinstance(self.vector_store, LocalVectorStore):
            return self.vector_store.indexed_embeddings
        return []

    @property
    def embed_id_to_doc_id(self):
        """Backward-compat: return embed_id mapping for local backend only."""
        if isinstance(self.vector_store, LocalVectorStore):
            return self.vector_store.embed_id_to_doc_id
        return {}

    @property
    def qdrant_client(self):
        """Backward-compat: expose qdrant client for tests that inspect it."""
        from .vector_store.qdrant import QdrantVectorStore
        if isinstance(self.vector_store, QdrantVectorStore):
            return self.vector_store.client
        return None

    @property
    def qdrant_collection(self):
        """Backward-compat: expose qdrant collection name."""
        from .vector_store.qdrant import QdrantVectorStore
        if isinstance(self.vector_store, QdrantVectorStore):
            return self.vector_store.collection_name
        return None
