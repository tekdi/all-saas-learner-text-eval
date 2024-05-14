import base64
import io
from functools import lru_cache
from pydub import AudioSegment
from pydub.silence import detect_silence
import eng_to_ipa as p
from fuzzywuzzy import fuzz
import jiwer
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

english_phoneme = ["b","d","f","g","h","ʤ","k","l","m","n","p","r","s","t","v","w","z","ʒ","tʃ","ʃ","θ","ð","ŋ","j","æ","eɪ","ɛ","i:","ɪ","aɪ","ɒ","oʊ","ʊ","ʌ","u:","ɔɪ","aʊ","ə","eəʳ","ɑ:","ɜ:ʳ","ɔ:","ɪəʳ","ʊəʳ","i","u","ɔ","ɑ","ɜ","e","ʧ","o","y","a", "x", "c"]
anamoly_list = {}

def calculate_snr(audio, sr):
    n_fft = min(len(audio), 2048)  # Ensure n_fft does not exceed the length of the audio
    stft = librosa.stft(audio, n_fft=n_fft)
    power = np.abs(stft)**2

    mel_spectrogram = librosa.feature.melspectrogram(S=power, sr=sr)
    mel_power = np.mean(mel_spectrogram, axis=0)

    energy_threshold = np.mean(mel_power)
    speech_indices = mel_power > energy_threshold
    noise_indices = ~speech_indices

    signal_power = np.mean(power[:, speech_indices], axis=1)
    average_signal_power = np.mean(signal_power) if signal_power.size > 0 else 0

    noise_power = np.mean(power[:, noise_indices], axis=1)
    average_noise_power = np.mean(noise_power) if noise_power.size > 0 else 1e-10

    snr = 10 * np.log10(average_signal_power / average_noise_power) if average_signal_power > 0 else 0
    return snr

def estimate_noise_floor(audio, sr, frame_length=None, hop_length=512):
    frame_length = frame_length or min(len(audio), 2048)
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    power_spectrogram = np.abs(stft)**2
    energy = np.sum(power_spectrogram, axis=0)

    low_energy_threshold = np.percentile(energy, 10)
    very_low_energy = energy[energy <= low_energy_threshold]

    adaptive_percentile = 5 if len(very_low_energy) < len(energy) * 0.1 else 10
    noise_floor = np.percentile(energy, adaptive_percentile)

    return noise_floor

def denoise_audio(filepath, speed_factor=1.0):
    audio, sample_rate = librosa.load(filepath, sr=None)

    # Apply time stretching first if the speed factor is not 1.0
    if speed_factor != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=speed_factor)

    # Calculate initial full audio SNR
    initial_snr = calculate_snr(audio, sample_rate)

    # Improved VAD
    vad_intervals = librosa.effects.split(audio, top_db=20)
    noise_floor = estimate_noise_floor(audio, sample_rate)

    noise_reduced_audio = np.copy(audio)
    improved_intervals = False  # Flag to track if any intervals improved SNR

    for interval in vad_intervals:
        interval_audio = audio[interval[0]:interval[1]]
        interval_snr = calculate_snr(interval_audio, sample_rate)

        # Determine reduction intensity based on initial SNR
        reduction_intensity = determine_reduction_intensity(initial_snr)

        # Apply noise reduction
        reduced_interval_audio = nr.reduce_noise(y=interval_audio, sr=sample_rate, prop_decrease=reduction_intensity)

        # Calculate SNR after noise reduction
        reduced_interval_snr = calculate_snr(reduced_interval_audio, sample_rate)
        if reduced_interval_snr > interval_snr:
            noise_reduced_audio[interval[0]:interval[1]] = reduced_interval_audio
            improved_intervals = True
        else:
            print("No SNR improvement; keeping original audio for this interval.")

    # Calculate final SNR and decide which version to use based on SNR comparison
    final_snr = calculate_snr(noise_reduced_audio, sample_rate)
    if not improved_intervals or final_snr < initial_snr:
        final_snr = initial_snr  # Revert to original SNR if no improvement
        noise_reduced_audio = audio  # Revert to original audio

    normalized_audio = librosa.util.normalize(noise_reduced_audio)
    return normalized_audio, sample_rate, initial_snr, final_snr

def determine_reduction_intensity(snr):
    if snr < 10:
        return 0.7
    elif snr < 15:
        return 0.5  
    elif snr < 20:
        return 0.22
    elif snr >= 30:
        return 0.1
    return 0.1  # Default to the least aggressive reduction if no specific conditions are met

def convert_to_base64(audio_data, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='wav')
    buffer.seek(0)
    base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
    return base64_audio

def get_error_arrays(alignments, reference, hypothesis, base64string):
    insertion = []
    deletion = []
    substitution = []

    for chunk in alignments[0]:
        if chunk.type == 'insert':
            insertion.extend(
                list(range(chunk.hyp_start_idx, chunk.hyp_end_idx)))
        elif chunk.type == 'delete':
            deletion.extend(
                list(range(chunk.ref_start_idx, chunk.ref_end_idx)))
        elif chunk.type == 'substitute':
            refslice = slice(chunk.ref_start_idx, chunk.ref_end_idx)
            hyposlice = slice(chunk.hyp_start_idx, chunk.hyp_end_idx)

            substitution.append({
                "removed": hypothesis[hyposlice],
                "replaced": reference[refslice]
            })

    insertion_chars = [hypothesis[i] for i in insertion]
    deletion_chars = [reference[i] for i in deletion]

    # For count the pauses in audio files
    audio_data = base64.b64decode(base64string)

    # Use pydub to load the audio from the BytesIO object
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

    # Check if the audio is completely silent or empty
    silence_ranges = detect_silence(
        audio_segment, min_silence_len=500, silence_thresh=-40)

    if len(silence_ranges) == 1 and silence_ranges[0] == [0, len(audio_segment)]:
        pause_count = 0
    else:
        # Count pause occurrences
        pause_count = len(silence_ranges)

    return {
        'insertion': insertion_chars,
        'deletion': deletion_chars,
        'substitution': substitution,
        'pause_count': pause_count
    }

def find_closest_match(target_word, input_string):
    # Tokenize the input string into words
    words = input_string.lower().split()
    targ = target_word.lower()
    # Initialize variables to keep track of the best match
    best_match = None
    best_score = 0
    
    # Iterate through the words in the input string
    for word in words:
        similarity_score = fuzz.ratio(targ, word)
        
        # Update the best match if a higher score is found
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = word
    
    return best_match, best_score

@lru_cache(maxsize=None)
def split_into_phonemes(token):
    # Phoneme mapping for combined phonemes
    combined_phonemes = {
        "dʒ": "ʤ",
        "tʃ": "ʧ",
        "ɪəʳ": "ɪəʳ",
        "ʊəʳ": "ʊəʳ",
        "eɪʳ": "eɪ",
        "aɪ": "aɪ",
        "oʊ": "o",
        "ɔɪ": "ɔɪ",
        "aʊ": "aʊ",
        "eəʳ": "eəʳ",
        "ɑ:": "ɑ",
        "ɜ:ʳ": "ɜ:ʳ",
        "ɔ:": "ɔ:",
        "i:": "i",
    }
    
    # Set of characters to skip (stress marks, etc.)
    skip_chars = {"'", " ", "ˈ", "ˌ"}

    # Convert the english_phoneme list into a set for O(1) average-time complexity checks
    english_phoneme_set = set(english_phoneme)
    
    ph_list = []
    word_list = token.split()  # split by whitespace (space, tab, newline, etc.)

    for p in word_list:
        size = len(p)
        i = 0
        while i < size:
            if p[i] in skip_chars:
                i += 1
                continue

            # Check for combined phonemes first (3 then 2 characters long)
            if i + 3 <= size and p[i:i+3] in combined_phonemes:
                ph_list.append(combined_phonemes[p[i:i+3]])
                i += 3
            elif i + 2 <= size and p[i:i+2] in combined_phonemes:
                ph_list.append(combined_phonemes[p[i:i+2]])
                i += 2
            elif i + 1 <= size and p[i:i+1] in english_phoneme_set:
                ph_list.append(p[i:i+1])
                i += 1
            else:
                # Log an anomaly if the character isn't recognized
                ph_list.append(p[i])
                if p[i] not in anamoly_list:
                    anamoly_list[p[i]] = 1
                else:
                    anamoly_list[p[i]] += 1
                i += 1

    return ph_list

def identify_missing_tokens(orig_text, resp_text):
    # Splitting text into words
    orig_word_list = orig_text.lower().split()
    resp_word_list = resp_text.lower().split()
    
    # Initialize lists and dictionaries
    construct_word_list = []
    missing_word_list = []
    orig_phoneme_list = []
    construct_phoneme_list = []
    missing_phoneme_list = []
    construct_text = []
    
    # Precompute phonemes for response words for quick lookup
    resp_phonemes = {word: p.convert(word) for word in resp_word_list}
    print("resp_phoneme::", resp_phonemes)
    for word in orig_word_list:
        # Precompute original word phonemes
        p_word = p.convert(word)
        
        # Find closest match based on precomputed phonemes to avoid redundant calculations
        closest_match, similarity_score = find_closest_match(word, resp_text)
        
        # Check similarity and categorize word
        if similarity_score > 80:
            construct_word_list.append(closest_match)
            p_closest_match = resp_phonemes[closest_match]
            construct_phoneme_list.append(split_into_phonemes(p_closest_match))
            construct_text.append(closest_match)
        else:
            missing_word_list.append(word)
            p_word_phonemes = split_into_phonemes(p_word)
            missing_phoneme_list.append(p_word_phonemes)
        
        # Store original phonemes for each word
        orig_phoneme_list.append(split_into_phonemes(p_word))

    # Convert list of words to a single string
    construct_text = ' '.join(construct_text)

    # Efficiently deduplicate and flatten phoneme lists
    #orig_flatList = set(phoneme for sublist in orig_phoneme_list for phoneme in sublist)
    missing_flatList = set(phoneme for sublist in missing_phoneme_list for phoneme in sublist)
    construct_flatList = set(phoneme for sublist in construct_phoneme_list for phoneme in sublist)

    return list(construct_flatList), list(missing_flatList) ,construct_text

def processLP(orig_text, resp_text):
    cons_list, miss_list, construct_text = identify_missing_tokens(orig_text, resp_text)

    #remove phonemes from miss_list which are in cons_list, ?but add those phonemes a count of could be issue

    # phonemes in constructed list are familiar ones
    # phonemes that are in miss_list and not in cons_list are the unfamiliar ones
    unfamiliar_list = []
    for c in miss_list:
        if c not in cons_list:
            unfamiliar_list.append(c)
    #function to calculate wer cer, substitutions, deletions and insertions, silence, repetitions
    #insert into DB the LearnerProfile vector
    return cons_list, miss_list,construct_text