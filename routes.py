import base64
from io import BytesIO
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from utils import convert_to_base64, denoise_audio, get_error_arrays, split_into_phonemes, processLP
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

    wer = jiwer.wer(reference, hypothesis)

    confidence_char_list =[]
    missing_char_list =[]
    construct_text=""

    if language == "en":
       confidence_char_list, missing_char_list,construct_text = processLP(reference,hypothesis)

    return {
        "wer": wer,
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
    reference = data.reference
    hypothesis = data.hypothesis
    base64_string = data.base64_string
    
    charOut = jiwer.process_characters(reference, hypothesis)

    # Extract error arrays
    error_arrays = get_error_arrays(
        charOut.alignments, reference, hypothesis, base64_string)

    if data.base64_string:
        audio_base64_string = data.base64_string
        if audio_base64_string:
            # Convert base64 audio to audio data
            audio_data = base64.b64decode(audio_base64_string)
            audio_io = BytesIO(audio_data)

            # Proceed with existing process
            denoised_audio, sample_rate, initial_snr, final_snr = denoise_audio(audio_io, speed_factor=0.75)
            denoised_audio_base64 = convert_to_base64(denoised_audio, sample_rate)

            # Delete audio data from cache
            del audio_data
            del audio_io
 
            return {
                "denoised_audio_base64": denoised_audio_base64,
                "insertion": error_arrays['insertion'],
                "insertion_count": len(error_arrays['insertion']),
                "deletion": error_arrays['deletion'],
                "deletion_count": len(error_arrays['deletion']),
                "substitution": error_arrays['substitution'],
                "substitution_count": len(error_arrays['substitution']),
                "pause_count": error_arrays['pause_count'],
                "cer": charOut.cer,
                }
        else:
            return {"error": "Missing audio_base64 parameter."}
    else:
        return {"error": "No data received."}
