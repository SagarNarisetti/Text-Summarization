import re as re2
import numpy as np1
import pandas as pd
import string
from nltk.corpus import stopwords
import unicodedata




# Removal of the punctuation from words
def remove_punctuation(word):
    cleaned_list = [value for value in word if value not in string.punctuation]
    return ''.join(cleaned_list)


# Removal of punctuation from the text data
def remove_punctuation_from_text(text):
    cleaned_text = [remove_punctuation(value) for value in text]
    return ''.join(cleaned_text)


# Remove numericals from the text data
def remove_number_from_text(sequence):
    sequence = re2.sub('[0-9]+', '', sequence)
    return ' '.join(sequence.split())


# handling stop words
def remove_stopwords(sequence):
    stop_words = stopwords.words('english')
    sequence = sequence.split()
    word_sequence = [value for value in sequence if value not in stop_words]
    return ' '.join(word_sequence)


def handle_contractions(txt, contraction_map=None):
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contra_patt = re2.compile(f'({contractions_keys})', flags=re2.DOTALL)

    def expd_word(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        es = contraction_map.get(match)
        if not es:
            print(match)
            return match
        return es

    es = contra_patt.sub(expd_word, txt)
    es = re2.sub("'", "", es)
    return es


# Cleaning text
def clean_text(sequence):
    sequence = sequence.lower()
    sequence = remove_punctuation_from_text(sequence)
    sequence = remove_number_from_text(sequence)
    sequence = remove_stopwords(sequence)

    # hadling special symbols like hyphens
    sequence = re2.sub('–', '', sequence)
    sequence = ' '.join(sequence.split())  # removing `extra` white spaces

    # Removing unnecessary characters from text
    sequence = re2.sub("(\\t)", ' ', str(sequence)).lower()
    # replacing \\t values with empty spaces
    sequence = re2.sub("(\\r)", ' ', str(sequence)).lower()
    # replacing \\r values with spaces
    sequence = re2.sub("(\\n)", ' ', str(sequence)).lower()

    sequence = unicodedata.normalize('NFKD', sequence).encode('ascii', 'ignore').decode(
        'utf-8', 'ignore'
    )

    sequence = re2.sub("(--+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\.\.+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(~~+)", ' ', str(sequence)).lower()
    sequence = re2.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM',
                       str(sequence)).lower()
    sequence = re2.sub("(\+\++)", ' ', str(sequence)).lower()
    sequence = re2.sub("(__+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\s+.\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(sequence)).lower()

    sequence = re2.sub("(mailto:)", ' ', str(sequence)).lower()

    sequence = re2.sub(r"(\\x9\d)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\-\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(mailto:)", ' ', str(sequence)).lower()

    sequence = re2.sub("(\.\s+)", ' ', str(sequence)).lower()

    sequence = re2.sub("(\:\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(sequence)).lower()

    try:
        url = re2.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(sequence))
        repl_url = url.group(3)
        # handling url present in the data
        sequence = re2.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(sequence))
    except Exception as e:
        pass

    sequence = re2.sub("(\s+.\s+)", ' ', str(sequence)).lower()
    # handling (\s+.\s+) by replacing with blank space

    return sequence


def trim_text_and_summary(df, maximum_text_length, maximum_summary_length):
    ctd = np1.array(df['text'])
    cleaned_summary_data = np1.array(df['headlines'])

    mini_text = []
    mini_summary = []

    for val in range(len(ctd)):
        if len(ctd[val].split()) <= maximum_text_length and len(
                cleaned_summary_data[val].split()
        ) <= maximum_summary_length:
            mini_text.append(ctd[val])
            mini_summary.append(cleaned_summary_data[val])

    df = pd.DataFrame({'text': mini_text, 'summary': mini_summary})
    return df


# calculating the rare words and other metrics
def rare_words_metrics(tokenizer, threshold):
    ct = 0
    freq = 0
    tol_count = 0
    tf = 0

    for key, val in tokenizer.word_counts.items():
        tol_count = tol_count + 1
        tf = tf + val
        if (val < threshold):
            ct = ct + 1
            freq = freq + val
    percentage_of_rare_words = (ct / tol_count) * 100
    total_coverage_rarewords = (freq / tf) * 100
    print('   rare word metrics metrics :')
    print('                       count :', ct)
    print('                 total count :', tol_count)
    print('    percentage of rare words :', percentage_of_rare_words)
    print('  total coverage of rareword :', total_coverage_rarewords)
    print('Total frequency of rare word :', tf)
    return ct, tol_count


def sequence_to_summary(input_sequence, target_word_index=None, reverse_target_word_index=None, start_token=None,
                        end_token=None):
    new_string = ''
    for val in input_sequence:
        if (
                (val != 0 and val != target_word_index[start_token]) and
                (val != target_word_index[end_token])
        ):
            new_string = new_string + reverse_target_word_index[val] + ' '
    return new_string


def sequence_to_text(input_sequence, reverse_source_word_index):
    new_string = ''
    for val in input_sequence:
        if val != 0:
            new_string = new_string + reverse_source_word_index[val] + ' '
    return new_string


def max_length_percentage(data,number):
    d=0
    for i1 in data:
        if len(i1.split())<=number:
            d=d+1
    percentage=round(d/len(data),2)
    return percentage