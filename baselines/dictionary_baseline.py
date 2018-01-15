# -*-coding:utf8-*-

'''
Note that this code is a refactored version of https://raw.githubusercontent.com/clarinsi/redi/master/redi.py.

The code has two basic usages:
 - no translation model is yet precomputed -- provide input_train and target_train files
 - translation model is already present -- provide its location with --tm argument
'''

import kenlm
from six.moves import cPickle
import os
from collections import defaultdict
import time
import sys
import numpy as np

def get_uppers(tokens_lists):
    '''
    Returns indices of upper letters of strings contained within token_list.
    e.g. get_uppers(['Auto'], ['jED'], ['asd']) -> [[0], [1,2], []]
    '''
    uppers = []
    for tokens in tokens_lists:
        token_list_uppers = []
        for index, char in enumerate(tokens):
            if char.isupper():
                token_list_uppers.append(index)
        uppers.append(token_list_uppers)
    return uppers


def apply_uppers(uppers, token_list):
    for token_index, indices in enumerate(uppers):
        token = token_list[token_index]
        for index in indices:
            if index < len(token):
                token = token[:index] + token[index].upper() + token[index + 1:]
        token_list[token_index] = token
    return token_list


def generate_diacritics(token_list, lexicon, lm, tm_weight):
    uppers = get_uppers(token_list)

    # lower down incoming text
    token_list = [e.lower() for e in token_list]

    indices = []
    for index, token in enumerate(token_list):
        if token in lexicon:  # cannot generate diacritics for OOV
            if len(lexicon[token]) == 1:  # if only one variant with diacritics is applicable, use it
                token_list[index] = list(lexicon[token].keys())[0]
            else:
                # if no language model is provided, apply the most frequent translation
                if lm == None:
                    token_list[index] = max(lexicon[token].items(), key=lambda x: x[1])[0]
                else:
                    # otherwise store an index of word on which to use language model
                    indices.append(index)
    for index in indices:
        hypotheses = {}
        for hypothesis in lexicon[token_list[index]]:
            sent = ' '.join(token_list[:index] + [hypothesis] + token_list[index + 1:])
            hypotheses[hypothesis] = (1 - tm_weight) * lm.score(sent) + tm_weight * lexicon[token_list[index]][
                hypothesis]
        token_list[index] = sorted(hypotheses, key=lambda x: -hypotheses[x])[0]
    return apply_uppers(uppers, token_list)


def infer(input_file, tm, lm, tm_weight):
    result = []
    with open(input_file, 'r') as reader:
        for line in reader:
            token_list = generate_diacritics(line.strip().split(' '), tm, lm, tm_weight)
            result.append(' '.join(token_list))

    return result


def defaultdict_ctor():
    return defaultdict(int)


def calculate_translation_model(file_with_input_sentences, file_with_target_sentences, num_traning_sentences,
                                verbose=False):
    '''
    Calculates translation model, which is a dictionary in form:
        tm['word_without_diacritics']['diacritized_variant'] = log10(# diacritized_variant / sum (# possible_diacritizations))
    '''

    translation_model_dictionary = defaultdict(defaultdict_ctor)

    with open(file_with_input_sentences, 'r') as input_reader:
        with open(file_with_target_sentences, 'r') as target_reader:
            start_time = time.time()
            for i, (input_line, target_line) in enumerate(zip(input_reader, target_reader)):
                input_words, target_words = input_line.strip().split(' '), target_line.strip().split(' ')
                if len(input_words) != len(target_words):
                    print('Skipping, not same number of words on line.')
                    continue

                for input_word, target_word in zip(input_words, target_words):
                    translation_model_dictionary[input_word][target_word] += 1

                if verbose and (i % 1e6 == 0):
                    end_time = time.time()
                    print(i, (end_time - start_time) / 1e6, end_time - start_time)
                    start_time = end_time

                if i > num_traning_sentences:
                    break

    # create probabilities instead of counts
    for base_key, value in translation_model_dictionary.items():
        num_words = np.sum(list(translation_model_dictionary[base_key].values()))
        for target_key, target_key_occurence_count in translation_model_dictionary[base_key].items():
            translation_model_dictionary[base_key][target_key] = np.log10(target_key_occurence_count / num_words)

    if verbose:
        print('Translation model size: {}'.format(len(translation_model_dictionary)))

    return translation_model_dictionary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple diacritic restoration tool.')
    parser.add_argument('input_test', type=str, help='Path to test input sentences (to diacritize).')
    parser.add_argument('target_test', type=str, help='Path to test target sentences (gold data).')

    parser.add_argument('--input_train', type=str, default="", help='Path to train input sentences.')
    parser.add_argument('--target_train', type=str, default="", help='Path to train target sentences (gold data).')

    parser.add_argument('--tm', type=str, help='Path to translation model')
    parser.add_argument('--tm_weight', type=float, default=1.0, help='Translation model weight (in [0;1]).')
    parser.add_argument('--lm', type=str, help='Path to language model')

    parser.add_argument('--store_tm_dir', type=str, default="",
                        help="Save computed translation model to this directory.")

    parser.add_argument('-nts', '--num_training_sentences', default=-1, type=int,
                        help="Number of sentences to use for training the translation model. Default -1 refers to use all provided data.")

    parser.add_argument('-v', '--verbose', default=False, action='store_true', help="Verbose mode")
    args = parser.parse_args()

    verbose = args.verbose

    if (not args.tm) and (not args.input_train or not args.target_train):
        raise ValueError('Either translation model or train files must be provided.')

    tm_weight = args.tm_weight
    if tm_weight > 1 or tm_weight < 0:
        raise ValueError('tm_weight must be in [0;1]')

    tm = args.tm
    if not args.tm:
        num_training_sentences = sys.maxsize if args.num_training_sentences == -1 else args.num_training_sentences
        tm = calculate_translation_model(args.input_train, args.target_train, num_training_sentences, verbose)
    else:
        tm = cPickle.load(open(args.tm, "rb"))

    if args.store_tm_dir:
        with open(os.path.join(args.store_tm_dir, 'tm.pkl'), 'wb') as f:
            cPickle.dump(tm, f)

    lm = None
    if args.lm:
        lm = kenlm.Model(args.lm)

    diacritized_sentences = infer(args.input_test, tm, lm, tm_weight)

    # load test target sentences and compute word accuracy
    with open(args.target_test, "r") as target_reader:
        target_sentences = target_reader.read().splitlines()

    with open('system_out.txt', 'w') as writer:
        for line in diacritized_sentences:
            writer.write(line)

    total_words, atotal_words = 0, 0 # all_words, words_with_at_least_one_alphanumerical_character
    words_correct, awords_correct = 0, 0
    for system_line, gold_line in zip(diacritized_sentences, target_sentences):
        for system_word, gold_word in zip(system_line.split(' '), gold_line.split(' ')):
            words_correct += system_word == gold_word
            total_words += 1

            is_alpha = False
            for c in gold_word:
                if c.isalpha():
                    is_alpha = True
                    break
            atotal_words += is_alpha
            awords_correct += is_alpha and system_word == gold_word

    print('Total words: {}'.format(total_words))
    print('Correctly diacritized: {}, {}'.format(words_correct, words_correct / total_words))

    # alphanumerical accuracy
    print('ATotal words: {}'.format(atotal_words))
    print('ACorrectly diacritized: {}, {}'.format(awords_correct , awords_correct / atotal_words))
