#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import argparse
import sys
import io

import diacritization_stripping_data

# percentage of words in the language with diacritics
word_diacritics_rate_dict = {
    'vi': 88.4,  # Vietnamese
    'ro': 31.0,  # Romanian
    'lv': 47.7,  # Latvian
    'cs': 52.5,  # Czech
    'sk': 41.4,  # Slovak
    'ga': 29.5,  # Irish
    'fr': 16.7,  # French
    'hu': 50.7,  # Hungarian
    'pl': 36.9,  # Polish
    'sv': 26.4,  # Swedish
    'pt': 13.3,  # Portuguese
    'gl': 13.3,  # Galician
    'et': 19.7,  # Estonian
    'es': 11.3,  # Spanish
    'nn': 12.8,  # Norwegian-Nynorsk
    'tr': 30.0,  # Turkish
    'ca': 11.1,  # Catalan
    'sl': 14.0,  # Slovenian
    'fi': 23.5,  # Finnish
    'nb': 11.7,  # Norwegian-Bokmaal
    'da': 10.2,  # Danish
    'de': 8.3,  # German
    'hr': 16.7  # Croatian
}

parser = argparse.ArgumentParser()
parser.add_argument("out_file", type=str, help="Path to the file to store sentences without diacritics")
parser.add_argument("language", type=str, help="Language of the incoming text (shorter format -- e.g. cs, de).")
parser.add_argument("--uninorms", action="store_true",
                    help="Use diacritization stripping based on Unicode Normalization")
parser.add_argument("--uninames", action="store_true",
                    help="Use diacritization stripping based on Unicode Names")
parser.add_argument("--diacritics_percentage", action="store_true", default=False,
                    help="Each line must have at least given amount of words with diacritics (see word_diacritics_rate)")
parser.add_argument("--out_input_file", type=str, help="Accompanies diacritics_percentage.")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Also compute statistics of the stripping")
args = parser.parse_args()

language = args.language

maps = []
if args.uninames: maps.append(diacritization_stripping_data.strip_diacritization_uninames)
if args.uninorms: maps.append(diacritization_stripping_data.strip_diacritization_uninorms)

def strip_diacritics(line):
    output = ""
    for c in line:
        for m in maps:
            if c in m:
                output += m[c]
                break
        else:
            output += c
    return output

with io.open(args.out_file, 'w', encoding='utf8') as writer:
    if args.diacritics_percentage:
        if language not in word_diacritics_rate_dict.keys():
            raise IOError('Provided language is not supported: {}'.format(language))
        required_words_with_diacritics_rate = word_diacritics_rate_dict[language]

        with io.open(args.out_input_file, 'w', encoding='utf8') as input_writer:
            for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8'):
                output = strip_diacritics(line)

                # check minimal amount of diacritics
                words_no_diacritics = output.split(' ')
                words_diacritics = line.split(' ')
                words_with_diacritics_rate = 100 - 100.0 * sum(
                    [words_no_diacritics[i] == words_diacritics[i] for i in range(len(words_diacritics))]) / len(
                    words_diacritics)
                if words_with_diacritics_rate < required_words_with_diacritics_rate:
                    continue

                input_writer.write(line)
                writer.write(output)
    else:
        for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8'):
            writer.write(strip_diacritics(line))
