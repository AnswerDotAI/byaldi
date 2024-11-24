from typing import Optional


class Result:
    def __init__(
        self,
        doc_id: str,
        chunk_id: int,
        score: float,
        metadata: Optional[dict] = None,
        text_chunk: Optional[str] = None,
    ):
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.score = score
        self.metadata = metadata or {}
        self.chunk = text_chunk

    def dict(self):
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "metadata": self.metadata,
            "chunk": self.chunk,
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()
