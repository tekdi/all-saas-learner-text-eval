from pydantic import BaseModel
from typing import List, Optional

class TextData(BaseModel):
    reference: str
    hypothesis: str
    language: str

class audioData(BaseModel):
    base64_string: str
    enablePauseCount:bool
    enableDenoiser:bool

class PhonemesRequest(BaseModel):
    text: str

class PhonemesResponse(BaseModel):
    phonemes: List[str]

class ErrorArraysResponse(BaseModel):
    cer: float
    insertion: List[str]
    insertion_count: int
    deletion: List[str]
    deletion_count: int
    substitution: List[dict]
    substitution_count: int
    pause_count: int

