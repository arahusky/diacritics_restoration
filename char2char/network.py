#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
import sys
import logging

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.metrics as tf_metrics

from common import utils


class Network:
    def __init__(self, input_alphabet_size, target_alphabet_size, cell_type, num_layers, rnn_cell_dim, embedding_dim,
                 logdir, expname, timestamp, learning_rate=1e-4, use_residual_connections=False, lm_loss_weigth=0,
                 threads=1, seed=42):

        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))
        # self.summary_writer = tf.summary.FileWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            cell_fn = utils.rnn_string_to_func(cell_type)

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.input_sentences = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_sentence_len_in_batch]
            self.sentence_lens = tf.placeholder(tf.int32, [None])  # [batch_size]
            self.target_sentences = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_sentence_len_in_batch]

            self.keep_prob = tf.placeholder_with_default(1.0, [], name='keep_prob')  # dropout keep probability

            max_sentence_len_in_batch = tf.shape(self.input_sentences)[1]

            input_words = None
            if embedding_dim < 1:
                input_words = tf.one_hot(self.input_sentences, input_alphabet_size)
            else:
                embedding_variables = tf.get_variable("embedding_variables", shape=[input_alphabet_size, embedding_dim])
                self.input_words_embedded = tf.nn.embedding_lookup(embedding_variables, self.input_sentences)
                input_words = tf.nn.dropout(self.input_words_embedded, self.keep_prob)

            next_to_last_layer_outputs = None

            cell_inputs = input_words
            for layer_index in range(num_layers):
                with tf.variable_scope('rnn_cell{}'.format(layer_index)):

                    fw_cell = cell_fn(rnn_cell_dim)
                    bw_cell = cell_fn(rnn_cell_dim)
                    (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                                  inputs=cell_inputs,
                                                                                  sequence_length=self.sentence_lens,
                                                                                  dtype=tf.float32)

                    layer_outputs = outputs_fw + outputs_bw
                    # TODO relu here???

                    # residual connection
                    if use_residual_connections:
                        if embedding_dim != rnn_cell_dim:
                            raise ValueError('Set embedding_dim == rnn_cell_dim to use residual connections!')
                        layer_outputs = layer_outputs + cell_inputs

                    cell_inputs = tf.nn.dropout(layer_outputs, self.keep_prob)

                    if num_layers > 1 and layer_index == num_layers - 2:
                        next_to_last_layer_outputs = cell_inputs

            outputs = cell_inputs  # [batch_size, max_sentence_len_in_batch, 2 * rnn_cell_dim]

            mask = tf.sequence_mask(self.sentence_lens, maxlen=max_sentence_len_in_batch)
            masked_outputs = tf.boolean_mask(outputs,
                                             mask)  # [maximally batch_size * max_sentence_len_in_batch, 2 * rnn_cell_dim]

            output_layer = tf_layers.fully_connected(masked_outputs, target_alphabet_size,
                                                     activation_fn=tf.nn.relu)  # [maximally batch_size * max_sentence_len_in_batch, alphabet_size]

            self.outputs_softmax = tf.nn.softmax(output_layer)

            self.predictions = tf.cast(tf.argmax(output_layer, 1),
                                       tf.int32)  # [maximally batch_size * max_sentence_len_in_batch, 1]

            # LM loss start
            loss_lm = 0
            if num_layers > 1 and lm_loss_weigth > 0:
                # we want to fetch all valid outputs but the last one (there is nothing to predict)
                lm_mask = tf.sequence_mask(self.sentence_lens - 1)
                next_to_last_layer_outputs_masked = tf.boolean_mask(next_to_last_layer_outputs, lm_mask)
                lm_predictions = tf_layers.fully_connected(next_to_last_layer_outputs_masked, target_alphabet_size)

                input_sentence_shifted = self.input_sentences[:, 1:]
                targets_lm_masked = tf.boolean_mask(input_sentence_shifted, lm_mask)
                loss_lm = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_predictions, labels=targets_lm_masked))

            # LM loss end

            targets_masked = tf.boolean_mask(self.target_sentences, mask)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer,
                                                                                 labels=targets_masked)) + lm_loss_weigth * loss_lm

            self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, self.global_step)
            # self.accuracy = tf_metrics.accuracy(self.predictions, targets_masked)
            #
            # logging.info('9')
            # self.dataset_name = tf.placeholder(tf.string, [])
            #
            # self.train_summary = tf.summary.scalar("train/accuracy", self.accuracy)
            # self.dev_summary = tf.summary.scalar("dev/accuracy", self.accuracy)
            # self.test_summary = tf.summary.scalar("test/accuracy", self.accuracy)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver(max_to_keep=None)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentences, sentence_lens, labels, keep_prob):
        _ = self.session.run([self.training],  # self.accuracy, self.train_summary],
                             {self.input_sentences: sentences,
                              self.sentence_lens: sentence_lens,
                              self.target_sentences: labels,
                              # self.dataset_name: "train",
                              self.keep_prob: keep_prob})
        # self.summary_writer.add_summary(summary, self.training_step)
        # return accuracy

    # def evaluate(self, sentences, sentence_lens, labels, dataset_name):
    #     if dataset_name == 'dev':
    #         accuracy, summary = self.session.run([self.accuracy, self.dev_summary],
    #                                              {self.input_sentences: sentences, self.sentence_lens: sentence_lens,
    #                                               self.target_sentences: labels, self.dataset_name: dataset_name})
    #     else:
    #         accuracy, summary = self.session.run([self.accuracy, self.test_summary],
    #                                              {self.input_sentences: sentences, self.sentence_lens: sentence_lens,
    #                                               self.target_sentences: labels, self.dataset_name: dataset_name})
    #     self.summary_writer.add_summary(summary, self.training_step)
    #     return accuracy

    def restore(self, checkpoint):
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            # print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))
