#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import datetime
import io
import logging
import os
import sys
import time
import tensorflow as tf

import numpy as np
from six.moves import cPickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parallelsentences_chars
from common import utils
from network import Network
from common import metrics

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp_name", default='', type=str, help="Experiment name.")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
    parser.add_argument("--embedding", default=200, type=int,
                        help="Embedding dimension. One hot is used if <1. It is highly recomended that embedding == rnn_cell_dim")

    parser.add_argument("--dataset", default='', type=str,
                        help="Path to dataset configuration file storing files for train, dev and test.")
    parser.add_argument("--input_char_vocab", default='', type=str,
                        help="Path to file storing input char vocabulary. If no provided, is automatically computed from data.")
    parser.add_argument("--target_char_vocab", default='', type=str,
                        help="Path to file storing target char vocabulary. If no provided, is automatically computed from data.")
    parser.add_argument("--num_top_chars", default=-1, type=int,
                        help="Take only num_top_chars most occuring characters. All other will be considered UNK")
    parser.add_argument('--use_additive_targets', action='store_true', default=False,
                        help="Default targets are direct letters, this argument sets targets to be like 'add colon above'")

    parser.add_argument("--max_chars", default=100, type=int, help="Maximum number of characters in a sentence.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--savedir", default="save", type=str, help="Savedir name.")

    parser.add_argument("--train_perc", default=1.0, type=float, help="Percentage of total samples used for training.")
    parser.add_argument("--validation_perc", default=0.0, type=float,
                        help="Percentage of total samples used for validation set.")
    parser.add_argument("--test_perc", default=0.0, type=float, help="Percentage of total samples used for testing.")

    parser.add_argument("--keep_prob", default=0.8, type=float, help="Dropout keep probability used for training.")

    parser.add_argument("--rnn_cell", default="gru", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=200, type=int, help="RNN cell dimension.")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of layers.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

    parser.add_argument('--use_residual', action='store_true', default=False,
                        help="If set, residual connections will be used in the char2char model.")
    parser.add_argument('--lm_loss_weight', default=0, type=float, help="Language model loss weight.")

    parser.add_argument("--save_every", default=1000, type=int, help="Interval for saving models.")
    parser.add_argument("--log_every", default=500, type=int, help="Interval for logging models (Tensorboard).")
    parser.add_argument("--num_test", default=1000, type=int, help="Number of samples to test on.")

    # arguments for various experiments
    parser.add_argument("--num_sentences", default=-1, type=int,
                        help="Number of sentences to read from train file (-1 == read all sentences).")

    args = parser.parse_args()

    experiment_name = args.exp_name
    experiment_name += '_layers{}_dim{}_embedding{}_lr{}'.format(args.num_layers, args.rnn_cell_dim,
                                                                 args.embedding, args.learning_rate)

    if args.dataset == '':
        raise ValueError('No dataset provided. Use --dataset argument.')

    # create save directory for current experiment's data (if not exists)
    save_data_dir = os.path.join(args.savedir, experiment_name)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # create subdir of save data directory to store trained models
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_model_dir = os.path.join(save_data_dir, timestamp)
    os.makedirs(save_model_dir)

    # configure logger
    logging.basicConfig(filename=os.path.join(save_model_dir, 'experiment_log.log'), level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info('Experiment started at: {} and its name: {}'.format(timestamp, experiment_name))
    logging.info('Experiment arguments: {}'.format(str(args)))

    dataset_files = utils.parse_dataset_file(args.dataset)
    print(dataset_files)
    # load train input and train target sentences
    print('Loading train data')
    input_sentences, target_sentences = [], []
    with io.open(dataset_files['train_inputs'], 'r', encoding='utf8') as reader:
        input_sentences = reader.read().splitlines()

    with io.open(dataset_files['train_targets'], 'r', encoding='utf8') as reader:
        target_sentences = reader.read().splitlines()
    # TODO refactor several lines up
    if args.num_sentences != -1:
        input_sentences = input_sentences[:args.num_sentences]
        target_sentences = target_sentences[:args.num_sentences]

    input_char_vocab, target_char_vocab = None, None
    if args.input_char_vocab != '':
        input_char_vocab = utils.load_vocabulary(args.input_char_vocab)
    if args.target_char_vocab != '':
        target_char_vocab = utils.load_vocabulary(args.target_char_vocab)

    use_additive_targets = args.use_additive_targets

    def czech_additive_targets_function(sentence, verbose = False):
        output = ''
        comma_set = {u'í', u'á', u'é', u'ý', u'ú', u'ó'}
        hook_set = {u'ž', u'ř', u'š', u'č', u'ď', u'ť', u'ň'}
        circle_set = {u'ů', u'ö'}

        for char in sentence:
            lowered_char = char.lower()
            if lowered_char in comma_set:
                output += '1'
            elif lowered_char in hook_set:
                output += '2'
            elif lowered_char in circle_set:
                output += '3'
            elif lowered_char.isspace():
                output += ' ' # keep space (required for metrics)
            else:
                output += '0'  # keep character

        return output

    if use_additive_targets:
        target_sentences = map(lambda x: czech_additive_targets_function(x), target_sentences)

    dataset = parallelsentences_chars.ParalelSentencesDataset(args.batch_size, args.max_chars, input_sentences,
                                                              target_sentences, args.train_perc, args.validation_perc,
                                                              args.test_perc, input_char_vocab, target_char_vocab,
                                                              args.num_top_chars)

    if 'dev_inputs' in dataset_files:
        print('Loading validation data')
        dev_input_sentences, dev_target_sentences = [], []
        with io.open(dataset_files['dev_inputs'], 'r', encoding='utf8') as reader:
            dev_input_sentences = reader.read().splitlines()

        with io.open(dataset_files['dev_targets'], 'r', encoding='utf8') as reader:
            dev_target_sentences = reader.read().splitlines()

        if use_additive_targets:
            dev_target_sentences = map(lambda x: czech_additive_targets_function(x), dev_target_sentences)

        dataset.add_validation_set(dev_input_sentences, dev_target_sentences)

    if 'test_inputs' in dataset_files:
        print('Loading test data')
        test_input_sentences, test_target_sentences = [], []
        with io.open(dataset_files['test_inputs'], 'r', encoding='utf8') as reader:
            test_input_sentences = reader.read().splitlines()

        with io.open(dataset_files['test_targets'], 'r', encoding='utf8') as reader:
            test_target_sentences = reader.read().splitlines()

        if use_additive_targets:
            test_target_sentences = map(lambda x: czech_additive_targets_function(x), test_target_sentences)

        dataset.add_test_set(test_input_sentences, test_target_sentences)

    print('Building dataset')
    dataset.build()

    input_char_vocab = dataset.input_char_vocabulary
    target_char_vocab = dataset.target_char_vocabulary

    # dump current configuration and used vocabulary to this model's folder
    with open(os.path.join(save_model_dir, 'vocab.pkl'), 'wb') as f:
        cPickle.dump((input_char_vocab, target_char_vocab), f)

    with open(os.path.join(save_model_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    evaluation_metrics = {'char_accuracy': metrics.c2c_per_char_accuracy,
                          'word_accuracy': metrics.c2c_per_word_accuracy}

    print('Creating network')
    # Construct the network
    network = Network(
        input_alphabet_size=len(input_char_vocab.keys()),
        target_alphabet_size=len(target_char_vocab.keys()),
        cell_type=args.rnn_cell,
        rnn_cell_dim=args.rnn_cell_dim,
        num_layers=args.num_layers,
        embedding_dim=args.embedding,
        logdir=args.logdir,
        expname=experiment_name,
        threads=args.threads,
        timestamp=timestamp,
        learning_rate=args.learning_rate,
        use_residual_connections=args.use_residual,
        lm_loss_weigth=args.lm_loss_weight
    )

    # split validation and testing data into "batches" (to fit into the memory)
    evaluation_sets = {}
    for evaluation_set_name, evaluation_set_fn in dataset.get_evaluation_sets():
        evaluation_set_sentences, evaluation_set_sentence_lens, evaluation_set_target_sentences = evaluation_set_fn()
        num_eval_samples = len(evaluation_set_sentences)
        num_eval_bins = np.ceil(num_eval_samples / dataset.batch_size)

        # print('{} : {} : {}'.format(evaluation_set_name, num_eval_samples, num_eval_bins))
        eval_set_input_sentences = np.array_split(evaluation_set_sentences, num_eval_bins)

        # print('{} : {} : {}'.format(evaluation_set_name, len(eval_set_input_sentences),
        #                             eval_set_input_sentences[0].shape))
        eval_set_sentence_lens = np.array_split(evaluation_set_sentence_lens, num_eval_bins)
        eval_set_target_sentences = np.array_split(evaluation_set_target_sentences, num_eval_bins)
        evaluation_sets[evaluation_set_name] = [eval_set_input_sentences, eval_set_sentence_lens,
                                                eval_set_target_sentences]

    summary_writer = tf.summary.FileWriter("{}/{}-{}".format(args.logdir, timestamp, experiment_name), flush_secs=10)

    # Train
    print('Training')
    for epoch in range(args.epochs):
        dataset.reset_batch_pointer()
        for batch_ind in range(dataset.num_batches):
            step_number = epoch * dataset.num_batches + batch_ind

            start = time.time()
            input_sentences, sentence_lens, target_sentences = dataset.next_batch()
            network.train(input_sentences, sentence_lens, target_sentences, args.keep_prob)
            end = time.time()

            print('Trained')

            if step_number % args.log_every == 0:
                eval_time_start = time.time()
                string_summary = "{}/{}, epoch: {}, time/batch = {:.3f}".format(step_number,
                                                                                args.epochs * dataset.num_batches,
                                                                                epoch, end - start)
                for eval_set_name in evaluation_sets.keys():
                    string_summary += "\n  {}".format(eval_set_name)
                    eval_set_input_sentences, eval_set_sentence_lens, eval_set_target_sentences = evaluation_sets[
                        eval_set_name]

                    eval_set_start_time = time.time()
                    predictions, lengths, targets = [], [], []
                    for eval_set_input_sentence, eval_set_sentence_len, eval_set_target_sentence in zip(
                            eval_set_input_sentences, eval_set_sentence_lens, eval_set_target_sentences):
                        eval_partial_set_result = network.session.run(network.predictions,
                                                                      feed_dict={
                                                                          network.input_sentences: eval_set_input_sentence,
                                                                          network.sentence_lens: eval_set_sentence_len,
                                                                          network.target_sentences: eval_set_target_sentence})

                        # eval_partial_set_result is a linearized list of dimension [ sum_i(eval_set_sentence_len[i])]
                        len_cumsum = np.cumsum(eval_set_sentence_len)[:-1]
                        eval_partial_set_result = np.array_split(eval_partial_set_result, len_cumsum)

                        print(len(eval_partial_set_result), len(eval_set_input_sentence), len(eval_set_sentence_len),
                              len(eval_set_target_sentence))

                        predictions += list(eval_partial_set_result)
                        lengths += list(eval_set_sentence_len)
                        targets += list(eval_set_target_sentence)

                    print(len(predictions), len(lengths), len(targets))
                    eval_set_middle_time = time.time()

                    summaries = []
                    for metric_name, metric_fn in evaluation_metrics.items():
                        metric_start_time = time.time()
                        result = metric_fn(predictions, lengths, targets, target_char_vocab)
                        metric_end_time = time.time()
                        string_summary += "\n    {}:{:.6f}:{}".format(metric_name, result,
                                                                      metric_end_time - metric_start_time)

                        summaries.append(
                            tf.Summary.Value(tag='{}_{}'.format(eval_set_name, metric_name), simple_value=result))

                    eval_set_end_time = time.time()

                    string_summary += "\n      {}:{}:{}".format(eval_set_name, eval_set_end_time - eval_set_start_time,
                                                                eval_set_middle_time - eval_set_start_time)

                    summary_writer.add_summary(tf.Summary(value=summaries), global_step=step_number)

                eval_time_end = time.time()
                string_summary += "\n Evaluation took : {}".format(eval_time_end - eval_time_start)

                logging.info(string_summary)

            if step_number % args.save_every == 0:
                checkpoint_path = os.path.join(save_model_dir, 'model.ckpt')
                network.saver.save(network.session, checkpoint_path, global_step=step_number)
