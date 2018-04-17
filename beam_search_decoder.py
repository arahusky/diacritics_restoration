# This code is a refactored version of UFAL Neural monkey decoder: https://github.com/ufal/neuralmonkey/blob/master/neuralmonkey/runners/singleton_beam_search_runner.py

from __future__ import print_function

import sys
import time

import numpy as np

from common import utils
from common import constants


def prepare_data(sentence, input_char_vocab):
    '''
    Converts given sentence to format appropriate for the neural model.
    '''
    inputs = np.zeros(len(sentence), dtype=np.int32)
    for i, char in enumerate(sentence):
        if char in input_char_vocab:
            input_token = input_char_vocab[char]
        else:
            input_token = input_char_vocab[constants.UNKNOWN_SYMBOL]

        inputs[i] = input_token

    input_lens = np.array([len(sentence)], np.int32)

    return inputs.reshape((1, -1)), input_lens


def evaluate_sentence_lm(language_model, hyp, vocabulary, original_sentence, eos=False):
    '''
    Computes language model probability of given hypothesis.
    '''
    # print(u''.join([utils.value_to_key(char, vocabulary) for char in tokens]))
    # print(tokens)

    sentence = hypothesis_to_sentence(hyp, vocabulary, original_sentence)

    lm_log_prob = language_model.score(sentence, eos=eos)

    # language model probability is normalized by number of words in the sentence
    normalized_lm_log_prob = lm_log_prob / (1 + len(sentence.split()))
    return normalized_lm_log_prob


def sort_hypotheses(hyps, gamma=0, normalized_lmscores=[]):
    """Sort hypotheses based on log probs and length.
    Args:
        hyps: A list of hypothesis.
        gamma: language model weight
        normalized_lmscores: language model scores for respective hypotheses
    Returns:
        hyps: A list of sorted hypothesis in reverse log_prob order.
    """

    if normalized_lmscores == []:
        return sorted(hyps, key=lambda h: np.sum(h.token_probabilities), reverse=True)

    normalized_hyp_probs = [np.sum(h.token_probabilities) / (len(h.token_probabilities) + 1) for h in
                            hyps]  # length normalized model log_probs
    rescores = [(1 - gamma) * p + gamma * l for (l, p) in
                zip(normalized_lmscores, normalized_hyp_probs)]  # combining model and language model log_probs

    hyps_with_probs_sorted = sorted(zip(hyps, rescores), key=lambda h: h[1], reverse=True)

    return list(map(lambda h: h[0], hyps_with_probs_sorted))


def hypothesis_to_sentence(hyp, target_char_vocabulary, original_sentence):
    '''
    Converts given hypothesis (integer indices) to sentence (characters).
    '''
    sentence = u''
    for i, char in enumerate(hyp.tokens):
        predicted_symbol = utils.value_to_key(char, target_char_vocabulary)
        if predicted_symbol == constants.UNKNOWN_SYMBOL:
            # if predicting unknown, copy corresponding input character
            sentence += original_sentence[i]
        else:
            sentence += predicted_symbol

    return sentence


class Hypothesis(object):
    """A class that represents a single hypothesis in a beam."""

    def __init__(self, tokens, token_probabilities):
        """Construct a new hypothesis object
        Arguments:
            tokens: The list of already decoded tokens
            token_probabilities: The log probabilities of the decoded tokens given the model
        """
        self.tokens = tokens
        self.token_probabilities = token_probabilities

    def extend(self, token, token_probability):
        """Return an extended version of the hypothesis.
        Arguments:
            token: The token to attach to the hypothesis
            token_probability: The log probability of emitting this token
        """
        return Hypothesis(self.tokens + [token], self.token_probabilities + [token_probability])

    @property
    def latest_token(self):
        """Get the last token from the hypothesis."""
        return self.tokens[-1]

    def __str__(self):
        return ("Hypothesis(log probs = {}, tokens = {})".format(
            self.token_probabilities, self.tokens))


class BeamSearchDecoder(object):
    """Implementation of simple beam search decoder. The beam search decoder is
    computed separately for each sentence in a batch so it's not as efficient
    as the batch solution. """

    def __init__(self, c2c_model, input_char_vocabulary, target_char_vocabulary, beam_size, language_model=None,
                 gamma_weight=0, whitespace_to_whitespace=True):
        """Construct a new instance of the runner.
        Arguments:
            c2c_model: Neural model used for decoding
            input_char_vocabulary: The vocabulary of decoder's inputs
            target_char_vocabulary: The vocabulary of decoder's targets
            beam_size: How many alternative hypotheses to run
            language_model: Language model to incorporate for weighting hypothesis, None if not use
            gamma_weight: after each word, sentence is evaluated with language model and its probability is added to current sentence probability with lambda_weight coefficient
            whitespace_to_whitespace: if True, all whitespaces in the generated sentence are the same whitespaces as in the source sentence, i.e. no whitespaces are added or removed.
             If False, there is no control above this (any character may become any character).
        """
        self.c2c_model = c2c_model
        self.input_char_vocabulary = input_char_vocabulary
        self.target_char_vocabulary = target_char_vocabulary
        self.beam_size = beam_size
        self.language_model = language_model
        self.whitespace_to_whitespace = whitespace_to_whitespace

        if ((self.language_model != None) and (gamma_weight == 0)):
            print('Zero weight was provided for the language model -- it will still be computed!')

        self.gamma_weight = gamma_weight

    def __call__(self, sess, dataset):
        """
        Arguments:
            sess: The session to use for computation
            dataset: The dataset to run the model on (list of sentences)
        """
        decoded_sentences = []

        for sentence_index, sentence in enumerate(dataset):
            inputs, input_lens = prepare_data(sentence, self.input_char_vocabulary)

            # print(u'Decoding sentence; {} with len {}'.format(sentence, len(sentence)))
            # print('Decoder max steps: {}'.format(decoder_max_step))
            start_time = time.time()

            feed = {self.c2c_model.input_sentences: inputs,
                    self.c2c_model.sentence_lens: input_lens,
                    self.c2c_model.keep_prob: 1.0}

            outputs_softmax = sess.run(self.c2c_model.outputs_softmax, feed_dict=feed)

            hyps = [Hypothesis([], [])]

            for char_index in range(len(outputs_softmax)):
                candidate_hyps = []

                for hyp in hyps:
                    character_probabilities = outputs_softmax[char_index]

                    if self.whitespace_to_whitespace:
                        # map whitespace character to whitespace character
                        if sentence[char_index].isspace():
                            if sentence[char_index] not in self.target_char_vocabulary:
                                candidate_hyps.append(
                                    hyp.extend(self.target_char_vocabulary[constants.UNKNOWN_SYMBOL], 0.0))
                            else:
                                candidate_hyps.append(
                                    hyp.extend(self.target_char_vocabulary[sentence[char_index]], 0.0))
                        else:
                            indices_sorted = np.argsort(character_probabilities)[
                                             ::-1]  # sort probabilities (descending order)

                            # until we create self.beam_size hypothesis or there are no more characters to process
                            hyps_added = 0
                            for best_ind in indices_sorted:
                                # sentence[char_index] is not whitespace; therefore no whitespace may be created from it
                                # TODO inverted vocabulary may perform better
                                if not utils.value_to_key(best_ind, self.target_char_vocabulary).isspace():
                                    log_prob = np.log(character_probabilities[best_ind])
                                    candidate_hyps.append(hyp.extend(best_ind, log_prob))
                                    hyps_added += 1

                                if hyps_added == self.beam_size:
                                    break

                    else:
                        k_best_indices = np.argpartition(character_probabilities, -self.beam_size)[-self.beam_size:]

                        # for each index in k_best indices, create a candidate hypothesis that extends "hyp".
                        for best_ind in k_best_indices:
                            log_prob = np.log(character_probabilities[best_ind])
                            candidate_hyps.append(hyp.extend(best_ind, log_prob))

                # if provided, incorporate language model; and sort hypothesis according to their log probability
                if self.language_model:
                    normalized_hyp_lm_probabilities = []
                    for hyp in candidate_hyps:
                        if sentence[char_index] == ' ' or char_index == len(sentence) - 1:
                            # TODO add LM eof flag
                            normalized_lm_probability = evaluate_sentence_lm(self.language_model, hyp,
                                                                             self.target_char_vocabulary,
                                                                             sentence)
                            # print(u'Result_: {}, log_prob: {}, log_prob_normalized: {}'.format(hypothesis_to_sentence(hyp, self.target_char_vocabulary), hyp.log_prob,
                            #                                                                    hyp.log_prob / len(hyp.tokens)))
                            # print('LM probability: {}'.format(lm_probability))

                            normalized_hyp_lm_probabilities.append(normalized_lm_probability)

                            # print(u'Result_: {}, log_prob: {}, log_prob_normalized: {}'.format(hypothesis_to_sentence(hyp, self.target_char_vocabulary), hyp.log_prob,
                            #                                                                    hyp.log_prob / len(hyp.tokens)))
                            # print('=================')

                    candidates_sorted = sort_hypotheses(candidate_hyps, self.gamma_weight,
                                                        normalized_hyp_lm_probabilities)
                else:
                    candidates_sorted = sort_hypotheses(candidate_hyps)

                # Create a new list of hypotheses that contains the best beam_size candidate hypotheses.
                hyps = candidates_sorted[:self.beam_size]

                # print('There are {} result hypothesis'.format(len(hyps)))
                # for hyp in hyps:
                #     print(
                #         u'Result_: {}, log_prob: {}, log_prob_normalized: {}'.format(hypothesis_to_sentence(hyp, self.target_char_vocabulary), hyp.log_prob,
                #                                                                      hyp.log_prob / len(hyp.tokens)))

            # Decode the sentence from indices to vocabulary. Remember, the
            # conversion vocabulary method wants a time x batch-shaped list
            end_time = time.time()
            sent = hypothesis_to_sentence(hyps[0], self.target_char_vocabulary, sentence)

            if sentence_index % 100 == 0:
                print(sentence_index, end_time - start_time)
                sys.stdout.flush()
                # print(u'Result sentence is: {}, infer time: {}'.format(sent, end_time - start_time))

            decoded_sentences.append(sent)

        return decoded_sentences
