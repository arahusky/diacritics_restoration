# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

from common import constants


class ParalelSentencesDataset():
    '''
    This class provides methods to load and batch dataset consisting of two parallel files: one containing input (uncorrected) sentences and the other with target
    (corrected) sentences.

    This class has two basic usage scenarios:
        - user has only one input and one target file and wants to create train, validation and test set (randomly) from them. In this case, instantiate this class with
        appropriate constants (train_perc, valid_perc and test_perc) and call build to prepare batches.
        - user has separate train, validation and test input and target files. In this case, ignore (set default values) those constants and use add_validation_set and
        use_test_set to add validation and test sets. After that, call build to prepare batches.

    '''

    def __init__(self, batch_size, max_chars_in_sentence, input_sentences, target_sentences, train_perc=1.0,
                 validation_perc=0.0, test_perc=0.0, input_char_vocabulary=None, target_char_vocabulary=None,
                 take_num_top_chars=-1):
        '''

        :param batch_size: how many samples should each training batch have.
        :param max_chars_in_sentence: maximal amount of characters each sentence might have to be included in the dataset.
        :param input_sentences: List with input (uncorrected) sentences.
        :param target_sentences: List containing target (corrected) sentences, where i-th item of this list contains corrected sentence of i-th item of data_input_file
        :param train_perc: percentage of input_data to use for training.
        :param validation_perc: percentage of input_data to use for validation.
        :param test_perc: percentage of input_data to use for testing.
        :param input_char_vocabulary: vocabulary (dictionary in form char:id) to use for encoding characters, None if new one should be created from the data.
        :param take_num_top_chars: take only this number of most occuring characters -- all other are considered UNK
        '''

        self.batch_size = batch_size
        self.max_chars_in_sentence = max_chars_in_sentence

        self.input_sentences = input_sentences
        self.target_sentences = target_sentences

        self.input_char_vocabulary = input_char_vocabulary
        self.target_char_vocabulary = target_char_vocabulary
        self.take_num_top_chars = take_num_top_chars

        self.train_perc = train_perc
        self.validation_perc = validation_perc
        self.test_perc = test_perc

        self.validation_input_sentences, self.validation_target_sentences = None, None
        self.test_input_sentences, test_targets = None, None

    def add_validation_set(self, validation_input_sentences, validation_target_sentences):
        self.validation_input_sentences = validation_input_sentences
        self.validation_target_sentences = validation_target_sentences

    def add_test_set(self, test_input_sentences, test_target_sentences):
        self.test_input_sentences = test_input_sentences
        self.test_target_sentences = test_target_sentences

    def build(self):
        if self.train_perc <= 0:
            print('No training samples!')
        if self.validation_perc == 0 and self.validation_input_sentences == None:
            print('No validation data!')
        if self.test_perc == 0 and self.test_input_sentences == None:
            print('No test data!')

        # first remove too long sentences and create vocabulary
        all_input_characters, all_target_characters = Counter(), Counter()
        num_input_sentences = len(self.input_sentences)
        self.input_sentences, self.target_sentences, train_input_characters, train_target_characters, num_removed = self._remove_long_samples_and_build_vocab(
            self.input_sentences, self.target_sentences)
        print('{}/{} train samples were removed due to their length.'.format(num_removed, num_input_sentences))
        all_input_characters.update(train_input_characters)
        all_target_characters.update(train_target_characters)

        if self.validation_input_sentences != None:
            self.validation_input_sentences, self.validation_target_sentences, validation_input_characters, validation_target_characters = self._build_vocab(
                self.validation_input_sentences, self.validation_target_sentences)
            all_input_characters.update(validation_input_characters)
            all_target_characters.update(validation_target_characters)

        if self.test_input_sentences != None:
            self.test_input_sentences, self.test_target_sentences, test_input_characters, test_target_characters = self._build_vocab(
                self.test_input_sentences, self.test_target_sentences)
            all_input_characters.update(test_input_characters)
            all_target_characters.update(test_target_characters)

        def build_vocab_from_counter(characters_counter):
            characters = list(sorted(characters_counter.keys()))

            if self.take_num_top_chars != -1:
                characters_tuples = characters_counter.most_common(self.take_num_top_chars)
                characters = map(lambda x: x[0], characters_tuples)
                characters = list(sorted(characters))

            char_vocabulary = {x: i for i, x in enumerate(characters)}
            char_vocabulary[constants.UNKNOWN_SYMBOL] = len(characters)

            return char_vocabulary

        # build vocabulary if no vocabulary is provided
        if self.input_char_vocabulary is None:
            self.input_char_vocabulary = build_vocab_from_counter(all_input_characters)

        if self.target_char_vocabulary is None:
            self.target_char_vocabulary = build_vocab_from_counter(all_target_characters)

        # self.char_vocab is a dictionary {"character" : ID}
        # characters is just a list of characters (words) that appeared in the text
        # characters = self.input_char_vocabulary.keys()
        # self.input_char_vocab_size = len(characters)

        # second step is transforming sentences into sequences of IDs rather than sequences of characters
        input_data, target_data, max_decoder_word_chars = self._preprocess(self.input_sentences, self.target_sentences)
        self.input_sentences, self.target_sentences = None, None  # forget no more necessary data
        self.max_decoder_word_chars = max_decoder_word_chars  # number of characters in the longest word

        if self.validation_input_sentences != None or self.test_input_sentences != None:
            self.train_xdata, self.train_ydata = input_data, target_data
            self.num_batches = int(len(input_data) / self.batch_size)  # number of train batches prepared

            if self.validation_input_sentences != None:
                self.validation_xdata, self.validation_ydata, valid_max_decoder_word_chars = self._preprocess(
                    self.validation_input_sentences, self.validation_target_sentences)
                self.validation_input_sentences, self.validation_target_sentences = None, None  # forget no more necessary data
                self.max_decoder_word_chars = max(self.max_decoder_word_chars, valid_max_decoder_word_chars)

            if self.test_input_sentences != None:
                self.test_xdata, self.test_ydata, test_max_decoder_word_chars = self._preprocess(
                    self.test_input_sentences, self.test_target_sentences)
                self.test_input_sentences, self.test_target_sentences = None, None  # forget no more necessary data
                self.max_decoder_word_chars = max(self.max_decoder_word_chars, test_max_decoder_word_chars)
        else:
            self._split_train_data(input_data, target_data, self.train_perc, self.validation_perc, self.test_perc)

        self.reset_batch_pointer()

    def _remove_long_samples_and_build_vocab(self, input_sentences, target_sentences):
        data_inputs_shortened, data_targets_shortened = [], []

        # remove too long (short) samples and split each sentence into words (split by space)
        num_samples_removed = 0
        input_characters = Counter()
        target_characters = Counter()
        for i in range(len(input_sentences)):
            if len(input_sentences[i]) < self.max_chars_in_sentence:
                data_inputs_shortened.append(input_sentences[i])
                data_targets_shortened.append(target_sentences[i])

                if self.input_char_vocabulary is None:
                    input_characters.update(input_sentences[i])
                if self.target_char_vocabulary is None:
                    target_characters.update(target_sentences[i])
            else:
                num_samples_removed += 1

        return data_inputs_shortened, data_targets_shortened, input_characters, target_characters, num_samples_removed

    def _build_vocab(self, input_sentences, target_sentences):
        data_inputs_shortened, data_targets_shortened = [], []

        # split each sentence into words (split by space)

        input_characters = Counter()
        target_characters = Counter()
        for i in range(len(input_sentences)):
            data_inputs_shortened.append(input_sentences[i])
            data_targets_shortened.append(target_sentences[i])

            if self.input_char_vocabulary is None:
                input_characters.update(input_sentences[i])
            if self.target_char_vocabulary is None:
                target_characters.update(target_sentences[i])

        return data_inputs_shortened, data_targets_shortened, input_characters, target_characters

    def _preprocess(self, input_sentences, target_sentences):

        max_decoder_word_chars = 0  # maximal number of characters in a single decoder word

        # encode data with the vocabulary
        # input_data(target_data) is a list of numpy arrays, where each numpy array is a sequence od char-IDs representing one word
        input_data = []
        target_data = []
        for input_sentence, target_sentence in zip(input_sentences, target_sentences):
            if len(input_sentence) != len(target_sentence):
                raise ValueError(
                    "Input and target sentence do not have same lengths!:\n input: {} \n target: {}".format(
                        input_sentence, target_sentence))

            input_data.append(np.array([self.input_char_vocabulary[char] if char in self.input_char_vocabulary
                                        else self.input_char_vocabulary[constants.UNKNOWN_SYMBOL] for char in
                                        input_sentence]))

            target_data.append(np.array([self.target_char_vocabulary[char] if char in self.target_char_vocabulary
                                         else self.target_char_vocabulary[constants.UNKNOWN_SYMBOL] for char in
                                         target_sentence]))

            if len(input_sentence) + 1 > max_decoder_word_chars:
                max_decoder_word_chars = len(input_sentence) + 1

        return input_data, target_data, max_decoder_word_chars

    def _split_train_data(self, input_data, target_data, train_percentage, valid_percentage, test_percentage):
        '''
        Splits input_data into train, validation and test sets
        '''

        num_available_batches = int(len(input_data) / self.batch_size)

        # counts in means of batches
        self.num_batches = int(round(num_available_batches * train_percentage))  # number of train batches prepared
        num_validation_batches = int(
            round(num_available_batches * valid_percentage))  # number of validation batches prepared
        num_test_batches = num_available_batches - self.num_batches - num_validation_batches  # number of test batches prepared

        # counts in means of samples
        num_test_samples = num_test_batches * self.batch_size
        num_validation_samples = num_validation_batches * self.batch_size
        num_train_samples = len(input_data) - num_test_samples - num_validation_samples

        if self.num_batches == 0:
            assert False, "Not enough data. Make batch_size smaller."

        print(str(self.num_batches) + ' train batches available')
        print(str(num_validation_batches) + ' validation batches available')
        print(str(num_test_batches) + ' test batches available')

        # split tensor into train, validation and test set
        self.test_xdata = input_data[:num_test_samples]
        self.test_ydata = target_data[:num_test_samples]

        self.validation_xdata = input_data[num_test_samples:num_test_samples + num_validation_samples]
        self.validation_ydata = target_data[num_test_samples:num_test_samples + num_validation_samples]

        self.train_xdata = input_data[num_test_samples + num_validation_samples:]
        self.train_ydata = target_data[num_test_samples + num_validation_samples:]

        assert len(self.train_xdata) == num_train_samples

    def next_batch(self):
        '''
        Returns next train batch.  Each batch consists of three parts.
        '''

        if self.pointer + self.batch_size > len(self.train_xdata):
            raise AssertionError('No more batches. Call reset_batch_pointer() before calling this method again.')

        max_sentence_len_in_batch = max([len(self.train_xdata[sentence_ind]) for sentence_ind in
                                         range(self.pointer, self.pointer + self.batch_size)])

        batch_inputs = np.zeros((self.batch_size, max_sentence_len_in_batch), np.int32)
        batch_targets = np.zeros((self.batch_size, max_sentence_len_in_batch), np.int32)
        batch_input_lens = np.zeros((self.batch_size), np.int32)

        for sample_ind in range(self.batch_size):
            batch_input_lens[sample_ind] = len(self.train_xdata[self.pointer])

            for char_ind in range(batch_input_lens[sample_ind]):
                batch_inputs[sample_ind][char_ind] = self.train_xdata[self.pointer][char_ind]
                batch_targets[sample_ind][char_ind] = self.train_ydata[self.pointer][char_ind]

            self.pointer += 1

        return batch_inputs, batch_input_lens, batch_targets

    def get_test_set(self, number_of_samples=-1):
        '''
        Returns numpy array with first number_of_samples test samples. If number_of_samples == -1, all test samples are returned.
        '''

        if number_of_samples == -1:
            number_of_samples = len(self.test_xdata)

        # firstly, prepare data storing words used in the current batch

        max_sentence_len_in_batch = max(
            [len(self.test_xdata[sentence_ind]) for sentence_ind in range(number_of_samples)])

        batch_inputs = np.zeros((number_of_samples, max_sentence_len_in_batch), np.int32)
        batch_targets = np.zeros((number_of_samples, max_sentence_len_in_batch), np.int32)
        batch_input_lens = np.zeros((number_of_samples), np.int32)

        for sample_ind in range(number_of_samples):
            batch_input_lens[sample_ind] = len(self.test_xdata[sample_ind])

            for char_ind in range(batch_input_lens[sample_ind]):
                batch_inputs[sample_ind][char_ind] = self.test_xdata[sample_ind][char_ind]
                batch_targets[sample_ind][char_ind] = self.test_ydata[sample_ind][char_ind]

        return batch_inputs, batch_input_lens, batch_targets

    def get_validation_set(self, number_of_samples=-1):
        '''
        Returns numpy array with first number_of_samples validation samples. If number_of_samples == -1, all validation samples are returned.
        '''

        if number_of_samples == -1:
            number_of_samples = len(self.validation_xdata)

        # firstly, prepare data storing words used in the current batch

        max_sentence_len_in_batch = max(
            [len(self.validation_xdata[sentence_ind]) for sentence_ind in range(number_of_samples)])

        batch_inputs = np.zeros((number_of_samples, max_sentence_len_in_batch), np.int32)
        batch_targets = np.zeros((number_of_samples, max_sentence_len_in_batch), np.int32)
        batch_input_lens = np.zeros((number_of_samples), np.int32)

        for sample_ind in range(number_of_samples):
            batch_input_lens[sample_ind] = len(self.validation_xdata[sample_ind])

            for char_ind in range(batch_input_lens[sample_ind]):
                batch_inputs[sample_ind][char_ind] = self.validation_xdata[sample_ind][char_ind]
                batch_targets[sample_ind][char_ind] = self.validation_ydata[sample_ind][char_ind]

        return batch_inputs, batch_input_lens, batch_targets

    def get_evaluation_sets(self):
        # return [('dev', self.get_validation_set)]
        return [('dev', self.get_validation_set), ('test', self.get_test_set)]

    def reset_batch_pointer(self):
        '''
        Resets batch pointer and permutes array. Call this after end of every epoch.
        '''
        permutation = np.random.permutation(len(self.train_xdata))

        # works only with np.arrays
        # self.train_xdata = self.train_xdata[permutation]
        # self.train_ydata = self.train_ydata[permutation]

        self.train_xdata = [self.train_xdata[i] for i in permutation]
        self.train_ydata = [self.train_ydata[i] for i in permutation]

        self.pointer = 0
