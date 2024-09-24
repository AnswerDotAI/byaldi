from typing import List
from pydantic import BaseModel


class ColPaliRequest(BaseModel):
    inputs: List[str]
    image_input: bool = False

