# -*- coding: utf-8 -*-
import io
import kenlm
import os
import sys

import tensorflow as tf
from six.moves import cPickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import beam_search_decoder
from network import Network


def infer(source_file, c2c_model_dir, language_model_path, beam_size, gamma_weight):
    # if not os.path.isfile(source_file):
    #    raise IOError('Provided source file "{}" does not exist.'.format(source_file))
    if not os.path.exists(c2c_model_dir):
        raise IOError('Provided experiment directory "{}" does not exist.'.format(c2c_model_dir))

    with io.open(source_file, 'r', encoding='utf8') as reader:
        sentences = reader.read().splitlines()

    with open(os.path.join(c2c_model_dir, 'vocab.pkl'), 'rb') as f:
        input_char_vocab, target_char_vocab = cPickle.load(f)

    with open(os.path.join(c2c_model_dir, 'config.pkl'), 'rb') as f:
        experiment_arguments = cPickle.load(f)

    use_residual = False
    if hasattr(experiment_arguments, 'use_residual'):
        use_residual = experiment_arguments.use_residual

    infer_model = Network(input_alphabet_size=len(input_char_vocab.keys()),
                          target_alphabet_size=len(target_char_vocab.keys()),
                          cell_type=experiment_arguments.rnn_cell,
                          num_layers=experiment_arguments.num_layers,
                          rnn_cell_dim=experiment_arguments.rnn_cell_dim,
                          embedding_dim=experiment_arguments.embedding,
                          logdir=None,
                          expname=None,
                          timestamp=None,
                          threads=8,  # TODO
                          use_residual_connections=use_residual
                          )

    sess = infer_model.session
    checkpoint = c2c_model_dir
    # tf.global_variables_initializer().run()
    saver = infer_model.saver
    ckpt = tf.train.get_checkpoint_state(checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise IOError('No model found in {}.'.format(c2c_model_dir))

    language_model = None
    if language_model_path != None:
        print('Loading language model')
        language_model = kenlm.Model(language_model_path)

    bsd = beam_search_decoder.BeamSearchDecoder(c2c_model=infer_model,
                                                vocabulary=vocabulary,
                                                beam_size=beam_size,
                                                language_model=language_model,
                                                gamma_weight=gamma_weight)

    corrected_sentences = bsd(sess, sentences)

    sess.close()

    return corrected_sentences


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=str, help="Path to file containing sentences to be corrected (one sentence per line).")
    parser.add_argument("dest_file", type=str, help="Path to file to store corrected output.")
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory with saved spell-checker model and configurations.")
    parser.add_argument("--lm", type=str, help="Path to trained language model.")
    parser.add_argument("--beam_size", default=1, type=int, help="Beam size used while decoding.")
    parser.add_argument("--alpha", default=0.0, type=float, help="Language model weight.")

    args = parser.parse_args()

    source_file = args.source_file
    exp_dir = args.exp_dir
    lm = args.lm
    beam_size = args.beam_size

    corrected_sentences = infer(source_file, exp_dir, lm, beam_size, args.alpha)

    with io.open(args.dest_file, 'w', encoding='utf8') as writer:
        for corrected_sentence in corrected_sentences:
            writer.write(u'{}\n'.format(corrected_sentence))
