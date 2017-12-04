# -*- coding: utf-8 -*-
import os
import sys
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infer

from common import evaluate_word_aligned

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="File containing sentences to be corrected (one sentence per line).")
    parser.add_argument("gold", type=str, help="File containing gold (corrected) sentences (one sentence per line).")
    parser.add_argument("model_dir", type=str, help="Path to experiment directory with saved spell-check model and configurations.")
    parser.add_argument("--lm", type=str, help="Path to experiment directory with saved language model and configurations.")
    parser.add_argument("--beam_size", default=1, type=int, help="Beam size used while decoding.")
    parser.add_argument("--gamma", default=0.0, type=float, help="Language model weight.")
    parser.add_argument("--input_evaluation", type=str, help="File passed to word evaluator as an input file. If none specified, 'input' is passed.")

    parser.add_argument("--corrected_file", type=str, help="File where to store corrected sentences (optional).")
    args = parser.parse_args()

    input_sentences = args.input
    gold_sentences = args.gold

    model_dir = args.model_dir
    lm_model_dir = args.lm
    beam_size = args.beam_size

    print('Beam size: {}, gamma: {}'.format(args.beam_size, args.gamma))

    corrected_sentences = infer.infer(input_sentences, model_dir, lm_model_dir, beam_size, args.gamma)

    if args.corrected_file is not None:
        # save corrected sentences
        with io.open(args.corrected_file, 'w', encoding='utf8') as writer:
            for line in corrected_sentences:
                writer.write(u'{}\n'.format(line))

    with io.open(input_sentences, 'r', encoding='utf8') as reader:
        input_sentences = reader.read().splitlines()

    with io.open(gold_sentences, 'r', encoding='utf8') as reader:
        gold_sentences = reader.read().splitlines()

    input_evaluation = input_sentences
    if args.input_evaluation:
        with io.open(args.input_evaluation, 'r', encoding='utf8') as reader:
            input_evaluation = reader.read().splitlines()

    accuracy, precision, recall, f1_score, total_words, gold_corrections = evaluate_word_aligned.evaluate_sentences(input_evaluation, gold_sentences, corrected_sentences)

    print("Words: {}, accuracy: {:.4f}%".format(total_words, accuracy))
    print("Gold corrections: {}, precision: {:.4f}%, recall: {:.4f}%, f-score: {:.4f}%".format(gold_corrections, precision, recall, f1_score))
