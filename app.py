import io
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import detect_silence
import jiwer
import eng_to_ipa as p
from fuzzywuzzy import fuzz

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
# Splitting text into words
    if resp_text == None:
        resp_text = ""
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

    for word in orig_word_list:
        # Precompute original word phonemes
        p_word = p.convert(word)

        # Find closest match based on precomputed phonemes to avoid redundant calculations
        closest_match, similarity_score = find_closest_match(word, resp_text)

        # Check similarity and categorize word
        if (closest_match != None) and (similarity_score >= 80 and len(orig_word_list) > 1) or (len(orig_word_list) == 1 and similarity_score >= 60):
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
    orig_flatList = set(phoneme for sublist in orig_phoneme_list for phoneme in sublist)
    #missing_flatList = set(phoneme for sublist in missing_phoneme_list for phoneme in sublist)
    construct_flatList = set(phoneme for sublist in construct_phoneme_list for phoneme in sublist)

    #For words like pew and few, we are adding to construct word and
    # we just need to eliminate the matching phonemes and
    # add missing phonemes into missing list
    for m in orig_flatList:
        if m not in construct_flatList:
            missing_phoneme_list.append(m)
    missing_flatList = set(phoneme for sublist in missing_phoneme_list for phoneme in sublist)
    return list(construct_flatList), list(missing_flatList),construct_text

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
