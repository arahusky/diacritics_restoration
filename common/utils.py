'''
Several convenient methods
'''

import io
import os

import numpy as np
from tensorflow.python.ops import rnn_cell

from . import constants


# from translation.custom_rnn_cells.layerNormGRUCell import LayerNormGRUCell as LayerNormGRUCell


def invert_vocabulary(vocabulary):
    inverted_vocab = {}
    for key, value in vocabulary.items():
        inverted_vocab[value] = key

    return inverted_vocab


def value_to_key(value, vocabulary):
    for key, val in vocabulary.items():
        if val == value:
            return key


def flatten_list_of_lists(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def load_vocabulary(filename):
    vocabulary = dict()
    with io.open(filename, encoding='utf-8', mode='r') as reader:
        keys = reader.read().splitlines()

    for i, key in enumerate(keys):
        vocabulary[key] = i

    return vocabulary


def parse_dataset_file(filename):
    dataset = dict()
    with io.open(filename, 'r', encoding='utf-8') as reader:
        for line in reader:
            key, value = line.strip().split(' ')
            dataset[key] = os.path.join(os.path.dirname(filename), value)

    return dataset


def rnn_string_to_func(rnn_string):
    if rnn_string == 'rnn':
        return rnn_cell.BasicRNNCell
    elif rnn_string == 'gru':
        return rnn_cell.GRUCell
    elif rnn_string == 'lstm':
        return rnn_cell.BasicLSTMCell
    # elif rnn_string == 'gru_ln':
    #    return LayerNormGRUCell
    else:
        raise Exception("model type not supported: {}".format(rnn_string))


def dataset_batch_to_readable_string(batch, vocabulary):
    '''

    :param batch: [batch_size, max_encoder_sentence_length]
    :param vocabulary:
    :return:
    '''
    sentences = []

    for sentence_ind in range(batch.shape[0]):
        sentence = []
        for char_ind in range(batch.shape[1]):
            char = batch[sentence_ind][char_ind]
            if char == vocabulary[constants.PAD_SYMBOL] or char == vocabulary[constants.EOS_SYMBOL]:
                break
            elif char == vocabulary[constants.GO_SYMBOL]:
                continue
            else:
                sentence.append(value_to_key(char, vocabulary))

        sentences.append(u''.join(sentence))

    return sentences


def decoder_outputs_to_sentences(decoder_outputs, vocabulary):
    '''

    :param decoder_outputs: [max_seq_len, batch_size, rnn_output_size]
    :param vocabulary:
    :return:
    '''

    sentences = []

    for batch_ind in range(decoder_outputs.shape[1]):
        sentence = []
        for time_ind in range(decoder_outputs.shape[0]):
            most_probable_char_ind = np.argmax(decoder_outputs[time_ind][batch_ind])
            sentence.append(most_probable_char_ind)

        sentences.append(sentence)

    return dataset_batch_to_readable_string(np.array(sentences), vocabulary)


def get_f_score(p, r, beta):
    return (1 + beta * beta) * p * r / (beta * beta * p + r)
