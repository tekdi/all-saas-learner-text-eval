import base64
import io
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from utils import convert_to_base64, denoise_audio, get_error_arrays, get_pause_count, split_into_phonemes, processLP
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
   # Convert base64 audio to audio data
  
   if data.base64_string:

    audio_base64_string = data.base64_string
    audio_data = base64.b64decode(audio_base64_string)
   
    pause_count = 0
    denoised_audio_base64 = ""

    if data.enablePauseCount:
        pause_count = get_pause_count(io.BytesIO(audio_data))

    if data.enableDenoiser:
        # Proceed with denoising process
        denoised_audio, sample_rate, initial_snr, final_snr = denoise_audio(io.BytesIO(audio_data), speed_factor=0.75)
        # Convert denoised audio to base64 string
        denoised_audio_base64 = convert_to_base64(denoised_audio, sample_rate)

    return {
        "denoised_audio_base64": denoised_audio_base64,
        "pause_count": pause_count
    }
