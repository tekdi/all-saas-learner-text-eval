import base64
import io
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from utils import  denoise_with_rnnoise, get_error_arrays, get_pause_count, split_into_phonemes, processLP
from schemas import TextData,audioData,PhonemesRequest, PhonemesResponse, ErrorArraysResponse
from typing import List
import jiwer
import eng_to_ipa as p

router = APIRouter()

@router.post('/getTextMatrices')
async def compute_errors(data: TextData):
    reference = data.reference
    hypothesis = data.hypothesis
    language = data.language

    charOut = jiwer.process_characters(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)

    confidence_char_list =[]
    missing_char_list =[]
    construct_text=""

    if language == "en":
       confidence_char_list, missing_char_list,construct_text = processLP(reference,hypothesis)

    # Extract error arrays
    error_arrays = get_error_arrays(
        charOut.alignments, reference, hypothesis)

    return {
        "wer": wer,
        "cer": charOut.cer,
        "insertion": error_arrays['insertion'],
        "insertion_count": len(error_arrays['insertion']),
        "deletion": error_arrays['deletion'],
        "deletion_count": len(error_arrays['deletion']),
        "substitution": error_arrays['substitution'],
        "substitution_count": len(error_arrays['substitution']),
        "confidence_char_list":confidence_char_list,
        "missing_char_list":missing_char_list,
        "construct_text":construct_text
    }


@router.post("/getPhonemes", response_model=dict)
async def get_phonemes(data: PhonemesRequest):
    phonemesList = split_into_phonemes(p.convert(data.text))
    return {"phonemes": phonemesList}


@router.post('/audio_processing')
async def audio_processing(data: audioData):
    audio_data = data.base64_string
    # Decode base64 to get the audio data
    audio_bytes = base64.b64decode(audio_data)
    audio_io = io.BytesIO(audio_bytes)

    pause_count = 0
    denoised_audio_base64 = ""

    if data.enablePauseCount:
        pause_count = get_pause_count(audio_io)  # Implement this function as needed

    if data.enableDenoiser:
        # If custom values are needed, specify here; otherwise, use defaults
        denoised_audio_base64 = denoise_with_rnnoise(audio_data, data.contentType)
         
        if denoised_audio_base64 is None:
            raise HTTPException(status_code=500, detail="Error during audio denoising")

    return {
        "denoised_audio_base64": denoised_audio_base64,
        "pause_count": pause_count
    }