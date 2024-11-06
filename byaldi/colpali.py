from typing import List, Union

import torch
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from PIL import Image


class ColPaliModel:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: str = "cuda",
        verbose: int = 1,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.verbose = verbose

        if "colpali" in pretrained_model_name_or_path.lower():
            self.model = ColPali.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if device == "cuda" else None,
            )
            self.processor = ColPaliProcessor.from_pretrained(self.pretrained_model_name_or_path)
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if device == "cuda" else None,
            )
            self.processor = ColQwen2Processor.from_pretrained(self.pretrained_model_name_or_path)
        else:
            raise ValueError("Unsupported model type. Use 'colpali' or 'colqwen2' models.")

        self.model = self.model.eval()
        if device != "cuda":
            self.model = self.model.to(device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, device: str = "cuda", verbose: int = 1):
        return cls(pretrained_model_name_or_path, device, verbose)

    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        with torch.no_grad():
            batch = self.processor.process_images(images)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            embeddings = self.model(**batch)

        return embeddings.cpu()

    def encode_query(self, query: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(query, str):
            query = [query]

        with torch.no_grad():
            batch = self.processor.process_queries(query)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            embeddings = self.model(**batch)

        return embeddings.cpu()

    def score(self, query: Union[str, List[str]], document_embeddings: List[torch.Tensor]) -> torch.Tensor:
        query_embeddings = self.encode_query(query)
        document_embeddings_tensor = torch.stack(document_embeddings)
        scores = self.processor.score(query_embeddings, document_embeddings_tensor)
        return scores.cpu()