import io
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import detect_silence
import jiwer
import eng_to_ipa as p
from fuzzywuzzy import fuzz
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

app = Flask(__name__)

anamoly_list={}

english_phoneme = ["b",
"d",
"f",
"g",
"h",
"ʤ",
"k",
"l",
"m",
"n",
"p",
"r",
"s",
"t",
"v",
"w",
"z",
"ʒ",
"tʃ",
"ʃ",
"θ",
"ð",
"ŋ",
"j",
"æ",
"eɪ",
"ɛ",
"i:",
"ɪ",
"aɪ",
"ɒ",
"oʊ",
"ʊ",
"ʌ",
"u:",
"ɔɪ",
"aʊ",
"ə",
"eəʳ",
"ɑ:",
"ɜ:ʳ",
"ɔ:",
"ɪəʳ",
"ʊəʳ",
"i",
"u",
"ɔ",
"ɑ",
"ɜ",
"e",
"ʧ",
"o",
"y",
"a", "x", "c"
]
@app.route('/audio_processing', methods=['POST'])
def home():
    data = request.json
    if data:
        audio_base64 = data.get('audio_base64')
        if audio_base64:
            # Convert base64 audio to audio data
            audio_data = base64.b64decode(audio_base64)
            audio_io = BytesIO(audio_data)

            # Proceed with existing process
            denoised_audio, sample_rate, initial_snr, final_snr = denoise_audio(audio_io, speed_factor=0.75)
            denoised_audio_base64 = convert_to_base64(denoised_audio, sample_rate)

            # Delete audio data from cache
            del audio_data
            del audio_io

            return jsonify({"denoised_audio_base64": denoised_audio_base64}), 200
        else:
            return jsonify({"error": "Missing audio_base64 parameter."}), 400
    else:
        return jsonify({"error": "No data received."}), 400

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

    # Initialize variables to keep track of the best match
    best_match = None
    best_score = 0

    # Iterate through the words in the input string
    for word in words:
        similarity_score = fuzz.ratio(target_word.lower(), word)

        # Update the best match if a higher score is found
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = word

    return best_match, best_score

def split_into_phonemes(token):
    ph_list = []
    spl_phone = False
    word_list = token.split()#split by space and .
    #now split using the ' notation
    for p in word_list:
        #print(ph_l)
        #ph_l = w.split("ˈ")
        #print(f"ph_l after split is: {ph_l}")
        #print(p[0])

        size = len(p)
        for i in range(size):
            #print(f"second for loop {p[i]}")
            if (p[i]=="'" or p[i] == " " or p[i]=="ˈ" or p[i]=="ˌ"):
                #it is just a notation for primary stress
                #print("found a special phoneme {spl_phone}")
                #do nothing
                print("just filler")
            elif (p[i] == "d" and size > i+1 and p[i+1] == "ʒ"):
                ph_list.append("ʤ")
                i = i+1
            elif (p[i]=="t" and size > i+1 and p[i+1]=="ʃ"):
                ph_list.append("tʃ")
                i = i+1
            elif (p[i]=="ɪ" and size > i+2 and p[i+1] == "ə" and p[i+2] == "ʳ"):
                ph_list.append("ɪəʳ")
                i = i+2
            elif (p[i]=="ʊ" and size > i+2 and p[i+1] == "ə" and p[i+2] == "ʳ"):
                ph_list.append("ʊəʳ")
                i = i+2
            elif (p[i]=="e" and size > i+2 and p[i+1]=="ɪ" and p[i+2]=="ʳ"):
                ph_list.append("eɪʳ")
                i = i+2
            elif (p[i]=="a" and size > i+1 and p[i+1]=="ɪ"):#aɪ
                ph_list.append("aɪ")
                i = i+1
            elif (p[i]=="o" and size > i+1 and p[i+1]=="ʊ"):#aɪ
                ph_list.append("oʊ")
                i = i+1
            elif (p[i]=="ɔ" and size > i+1 and p[i+1]=="ɪ"):#aɪ
                ph_list.append("ɔɪ")
                i = i+1
            elif (p[i]=="a" and size > i+1 and p[i+1]=="ʊ"):#aɪ
                ph_list.append("aʊ")
                i = i+1
            elif (p[i]=="e" and size > i+2 and p[i+1]=="ə" and p[i+2] == "ʳ"):#aɪ
                ph_list.append("eəʳ")
                i = i+2
            elif (p[i]=="ɑ" and size > i+1 and p[i+1]==":"):#aɪ
                ph_list.append("ɑ:")
                i = i+1
            elif (p[i]=="ɜ" and size > i+2 and p[i+1]==":" and p[i+2] == "ʳ"):#aɪ
                ph_list.append("ɜ:ʳ")
                i = i+2
            elif (p[i]=="ɔ" and size > i+1 and p[i+1]==":"):#aɪ
                ph_list.append("ɔ:")
                i = i+1
            elif (p[i]=="ɑ" and size > i+1 and p[i+1]==":"):#aɪ
                ph_list.append("ɑ:")
                i = i+1
            elif (p[i]=="ɪ" and size > i+2 and p[i+1]=="ə" and p[i+2] == "ʳ"):#ɪəʳ
                ph_list.append("ɪəʳ")
                i = i+2
            elif (p[i]=="ʊ" and size > i+2 and p[i+1]=="ə" and p[i+2]=="ʳ"):#ʊəʳ
                ph_list.append("ʊəʳ")
                i = i+2
            elif (p[i]=="i" and size > i+1 and p[i+1]==":"):
                ph_list.append("i:")
                i = i+1
            elif (p[i] in english_phoneme):
                ph_list.append(p[i])
            else:
                print(f"Not part of 44 phonemes: {p[i]}")
                if p[i] not in anamoly_list.keys():
                    anamoly_list[p[i]] = 1
                else:
                    count = anamoly_list[p[i]]
                    anamoly_list[p[i]] = count + 1 # add another count to the global dictionary
        #print(f"out of second loop {ph_list}")
    #print(f"phonemes for the word - {token} is {ph_list}")
    return ph_list

def identify_missing_tokens(orig_text, resp_text):
    if resp_text == None:
        resp_text = ""
    orig_word_list = orig_text.split()
    resp_word_list = resp_text.split()
    construct_word_list =[]
    missing_word_list=[]
    orig_phoneme_list = []
    construct_phoneme_list = []
    missing_phoneme_list =[]
    construct_text=''
    index=0
    for word in orig_word_list:
        #use similarity algo euclidean distance and add them, if there is no direct match
        closest_match, similarity_score = find_closest_match(word, resp_text)
        print(f"word:{word}: closest match: {closest_match}: sim score:{similarity_score}")
        p_word = p.convert(word)
        print(f"word - {word}:: phonemes - {p_word}")#p_word = split_into_phonemes(p_word)
        if closest_match != None and (similarity_score > 80 or len(orig_word_list) == 1):
            #print("matched word")
            construct_word_list.append(closest_match)
            p_closest_match = p.convert(closest_match)
            construct_phoneme_list.append(split_into_phonemes(p_closest_match))
            construct_text += closest_match + ' '
        else:
            print(f"no match for - {word}: closest match: {closest_match}: sim score:{similarity_score}")
            missing_word_list.append(word)
            missing_phoneme_list.append(split_into_phonemes(p_word))
        index = index+1
        orig_phoneme_list.append(split_into_phonemes(p_word))

        # iterate through the sublist using List comprehension to flatten the nested list to single list
        orig_flatList = [element for innerList in orig_phoneme_list for element in innerList]
        missing_flatList = [element for innerList in missing_phoneme_list for element in innerList]
        construct_flatList = [element for innerList in construct_phoneme_list for element in innerList]

        # ensure duplicates are removed and only unique set are available
        orig_flatList = list(set(orig_flatList))
        missing_flatList = list(set(missing_flatList))
        construct_flatList = list(set(construct_flatList))

        #For words like pew and few, we are adding to construct word and
        # we just need to eliminate the matching phonemes and
        # add missing phonemes into missing list
        for m in orig_flatList:
            print(m, " in construct phonemelist")
            if m not in construct_flatList:
                missing_flatList.append(m)
                print('adding to missing list', m)
        missing_flatList = list(set(missing_flatList))

        print(f"orig Text: {orig_text}")
        print(f"Resp Text: {resp_text}")
        print(f"construct Text: {construct_text}")

        print(f"original phonemes: {orig_phoneme_list}")
        #print(f"flat original phonemes: {orig_flatList}")
        print(f"Construct phonemes: {construct_phoneme_list}")

        #print(f"flat Construct phonemes: {construct_flatList}")
        #print(f"missing phonemes: {missing_phoneme_list}")
        print(f"flat missing phonemes: {missing_flatList}")
    return construct_flatList, missing_flatList,construct_text

def processLP(orig_text, resp_text):
    cons_list, miss_list,construct_text = identify_missing_tokens(orig_text, resp_text)
    print(f"constructed list:{cons_list}")
    print(f"missed list:{miss_list}")

    #remove phonemes from miss_list which are in cons_list, ?but add those phonemes a count of could be issue

    # phonemes in constructed list are familiar ones
    # phonemes that are in miss_list and not in cons_list are the unfamiliar ones
    unfamiliar_list = []
    for c in miss_list:
        if c not in cons_list:
            unfamiliar_list.append(c)
    print(f"Not Familiar with:{unfamiliar_list}")
    print( f"Anomaly list: {anamoly_list}")
    return cons_list, miss_list,construct_text
    #function to calculate wer cer, substitutions, deletions and insertions, silence, repetitions
    #insert into DB the LearnerProfile vector

@app.route('/getTextMatrices', methods=['POST'])
def compute_errors():
    data = request.get_json()
    reference = data.get('reference')
    hypothesis = data.get('hypothesis')
    base64_string = data.get('base64_string')
    language = data.get('language')

    charOut = jiwer.process_characters(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)

    confidence_char_list =[]
    missing_char_list =[]
    construct_text=""

    if language == "en":
       confidence_char_list, missing_char_list,construct_text = processLP(reference,hypothesis)

    # Extract error arrays
    error_arrays = get_error_arrays(
        charOut.alignments, reference, hypothesis, base64_string)

    return jsonify({
        "wer": wer,
        "cer": charOut.cer,
        "insertion": error_arrays['insertion'],
        "insertion_count": len(error_arrays['insertion']),
        "deletion": error_arrays['deletion'],
        "deletion_count": len(error_arrays['deletion']),
        "substitution": error_arrays['substitution'],
        "substitution_count": len(error_arrays['substitution']),
        "pause_count": error_arrays['pause_count'],
        "confidence_char_list":confidence_char_list,
        "missing_char_list":missing_char_list,
        "construct_text":construct_text
    })

@app.route('/getPhonemes', methods=['POST'])
def get_phonemes():
    data = request.get_json()
    text = data.get('text')

    phonemesList = split_into_phonemes(p.convert(text))

    return jsonify({
        "phonemes": phonemesList
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=False)
