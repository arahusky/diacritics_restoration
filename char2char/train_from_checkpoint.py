#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import io
import logging
import os
import sys
import time

import numpy as np
from six.moves import cPickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import parallelsentences_chars
from common import utils
from network import Network

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_dir", default='', type=str, help="Path to model dir storing its checkpoint.")
    parser.add_argument("--exp_name", default='', type=str, help="Experiment name.")
    parser.add_argument("--batch_size", default=200, type=int, help="Batch size.")

    parser.add_argument("--dataset", default='', type=str, help="Path to dataset configuration file storing files for train, dev and test.")

    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--savedir", default="save", type=str, help="Savedir name.")

    parser.add_argument("--keep_prob", default=0.8, type=float, help="Dropout keep probability used for training.")

    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--save_every", default=1000, type=int, help="Interval for saving models.")
    parser.add_argument("--log_every", default=100, type=int, help="Interval for logging models (Tensorboard).")

    args = parser.parse_args()

    model_dir = args.model_dir

    if not os.path.exists(model_dir):
        raise IOError('Provided model_dir does not exist!')

    with open(os.path.join(model_dir, 'vocab.pkl'), 'rb') as f:
        char_vocab = cPickle.load(f)

    with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
        experiment_arguments = cPickle.load(f)

    use_residual = False
    if hasattr(experiment_arguments, 'use_residual'):
        use_residual = experiment_arguments.use_residual

    batch_size = args.batch_size
    embedding = experiment_arguments.embedding
    dataset = args.dataset
    epochs = args.epochs
    logdir = args.logdir
    savedir = args.savedir
    keep_prob = args.keep_prob
    rnn_cell = experiment_arguments.rnn_cell
    rnn_cell_dim = experiment_arguments.rnn_cell_dim
    num_layers = experiment_arguments.num_layers
    learning_rate = args.learning_rate
    threads = args.threads
    use_residual = experiment_arguments.use_residual
    lm_loss_weight = experiment_arguments.lm_loss_weight
    save_every = args.save_every
    log_every = args.log_every
    max_chars = experiment_arguments.max_chars

    experiment_name = args.exp_name
    experiment_name += '_tfc_layers{}_dim{}_embedding{}_lr{}'.format(num_layers, rnn_cell_dim, embedding, learning_rate)

    if dataset == '':
        raise ValueError('No dataset provided. Use --dataset argument.')

    # create save directory for current experiment's data (if not exists)
    save_data_dir = os.path.join(savedir, experiment_name)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # create subdir of save data directory to store trained models
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_model_dir = os.path.join(save_data_dir, timestamp)
    os.makedirs(save_model_dir)

    # configure logger
    logging.basicConfig(filename=os.path.join(save_model_dir, 'experiment_log.log'), level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('Experiment started at: {} and its name: {}'.format(timestamp, experiment_name))
    logging.info('Experiment arguments: {}'.format(str(args)))

    dataset_files = utils.parse_dataset_file(dataset)
    print(dataset_files)
    # load train input and train target sentences
    print('Loading train data')
    input_sentences, target_sentences = [], []
    with io.open(dataset_files['train_inputs'], 'r', encoding='utf8') as reader:
        input_sentences = reader.read().splitlines()

    with io.open(dataset_files['train_targets'], 'r', encoding='utf8') as reader:
        target_sentences = reader.read().splitlines()

    dataset = parallelsentences_chars.ParalelSentencesDataset(batch_size, max_chars, input_sentences, target_sentences, char_vocabulary=char_vocab)

    if 'dev_inputs' in dataset_files:
        print('Loading validation data')
        dev_input_sentences, dev_target_sentences = [], []
        with io.open(dataset_files['dev_inputs'], 'r', encoding='utf8') as reader:
            dev_input_sentences = reader.read().splitlines()

        with io.open(dataset_files['dev_targets'], 'r', encoding='utf8') as reader:
            dev_target_sentences = reader.read().splitlines()

        dataset.add_validation_set(dev_input_sentences, dev_target_sentences)

    if 'test_inputs' in dataset_files:
        print('Loading test data')
        test_input_sentences, test_target_sentences = [], []
        with io.open(dataset_files['test_inputs'], 'r', encoding='utf8') as reader:
            test_input_sentences = reader.read().splitlines()

        with io.open(dataset_files['test_targets'], 'r', encoding='utf8') as reader:
            test_target_sentences = reader.read().splitlines()

        dataset.add_test_set(test_input_sentences, test_target_sentences)

    logging.info('Building dataset')
    dataset.build()

    # dump current configuration and used vocabulary to this model's folder
    with open(os.path.join(save_model_dir, 'vocab.pkl'), 'wb') as f:
        cPickle.dump(char_vocab, f)

    with open(os.path.join(save_model_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(experiment_arguments, f)

    logging.info('Creating network')
    # Construct the network
    network = Network(
        alphabet_size=dataset.char_vocab_size,
        cell_type=rnn_cell,
        rnn_cell_dim=rnn_cell_dim,
        num_layers=num_layers,
        embedding_dim=embedding,
        logdir=logdir,
        expname=experiment_name,
        threads=threads,
        timestamp=timestamp,
        learning_rate=learning_rate,
        use_residual_connections=use_residual,
        lm_loss_weigth=lm_loss_weight
    )

    # Restore the network from the last checkpoint
    logging.info('Restoring')
    checkpoint = model_dir
    saver = network.saver
    ckpt = tf.train.get_checkpoint_state(checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
        saver.restore(network.session, ckpt.model_checkpoint_path)
    else:
        raise IOError('No model found in {}.'.format(checkpoint))

    # Train
    logging.info('Training')

    for epoch in range(epochs):
        dataset.reset_batch_pointer()
        for batch_ind in range(dataset.num_batches):
            step_number = epoch * dataset.num_batches + batch_ind

            start = time.time()
            input_sentences, sentence_lens, target_sentences = dataset.next_batch()
            network.train(input_sentences, sentence_lens, target_sentences, keep_prob)
            end = time.time()

            if step_number % log_every == 0:
                input_sentences, sentence_lens, target_sentences = dataset.get_validation_set(1000)
                dev_accuracy = network.evaluate(input_sentences, sentence_lens, target_sentences, "dev")

                input_sentences, sentence_lens, target_sentences = dataset.get_test_set(1000)
                test_accuracy = network.evaluate(input_sentences, sentence_lens, target_sentences, "test")

                logging.info((
                    "{}/{}, epoch: {}, dev_acc:{:.6f}, test_acc:{:.6f}, time/batch = {:.3f}".format(step_number, epochs * dataset.num_batches, epoch, dev_accuracy,
                                                                                                    test_accuracy, end - start)))
            if step_number % save_every == 0:
                checkpoint_path = os.path.join(save_model_dir, 'model.ckpt')
                network.saver.save(network.session, checkpoint_path, global_step=step_number)
