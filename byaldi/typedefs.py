from pydantic import BaseModel
from typing import List, Optional, Dict


class ColPaliRequest(BaseModel):
    inputs: List[str]
    image_input: bool = False


class Result(BaseModel):
    doc_id: str
    page_num: int
    score: float
    metadata: dict = {}
    base64: str = None