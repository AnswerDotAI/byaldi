from typing import Optional


class Result:
    def __init__(self, doc_id: str, page_num: int, score: float, metadata: Optional[dict] = None, base64: Optional[str] = None):
        self.doc_id = doc_id
        self.page_num = page_num
        self.score = score
        self.metadata = metadata or {}
        self.base64 = base64

    def dict(self):
        return {
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "score": self.score,
            "metadata": self.metadata,
            "base64": self.base64,
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()
