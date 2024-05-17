from pydantic import BaseModel
from typing import List, Optional

class TextData(BaseModel):
    reference: str
    hypothesis: str
    base64_string: str
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
    wer: float
    cer: float
    insertion: List[str]
    insertion_count: int
    deletion: List[str]
    deletion_count: int
    substitution: List[dict]
    substitution_count: int
    pause_count: int
    confidence_char_list: Optional[List[str]]
    missing_char_list: Optional[List[str]]
    construct_text: Optional[str]
