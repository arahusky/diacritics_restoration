#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import argparse
import sys

# refactored version of Milan Straka's original code

def evaluate_sentences(original_sentences, gold_sentences, corrected_sentences, verbose = False):
    # Accuracy measurement
    acc_total, acc_correct = 0, 0
    # F-score measurement
    f_gold, f_system, f_both = 0, 0, 0

    if len(set([len(original_sentences), len(gold_sentences), len(corrected_sentences)])) != 1:
        raise ValueError("The given files do not contain the same number of lines")

    for i in range(len(original_sentences)):
        text, gold, system = original_sentences[i], gold_sentences[i], corrected_sentences[i]

        text, gold, system = map(lambda s: s.split(" "), [text, gold, system])
        if len(text) != len(gold) or len(gold) != len(system):
            print(text)
            print(gold)
            print(system)
            raise ValueError("The given files to not contain the same number of words on line '{}'".format(" ".join(text)))

        # Accuracy
        acc_total += len(gold)
        for i in range(len(gold)):
            if gold[i] == system[i]:
                acc_correct += 1
            else:
                # print(gold, system)
                if verbose:
                    print(gold[i].encode('utf8'), system[i].encode('utf8'))

        # F-score
        for i in range(len(gold)):
            if gold[i] != text[i]:
                f_gold += 1
                if gold[i] == system[i]:
                    f_both += 1
            if system[i] != text[i]:
                f_system += 1

    accuracy = 100 * acc_correct / acc_total
    precision = 100 * f_both / f_system
    recall = 100 * f_both / f_gold
    f1_score = 100 * 2 * f_both / (f_system + f_gold)
    total_words = acc_total
    gold_corrections = f_gold

    return accuracy, precision, recall, f1_score, total_words, gold_corrections


def evaluate(original_file, gold_file, system_file, verbose = False):
    # Accuracy measurement
    acc_total, acc_correct = 0, 0
    # F-score measurement
    f_gold, f_system, f_both = 0, 0, 0

    while True:
        text, gold, system = map(lambda f: f.readline(), [original_file, gold_file, system_file])
        if not (text) and not (gold) and not (system):
            break
        if not (text) or not (gold) or not (system):
            raise ValueError("The given files do not contain the same number of lines")

        text, gold, system = map(lambda s: s.rstrip("\r\n "), [text, gold, system])

        if len(text) != len(gold) or len(gold) != len(system):
            print(len(text), text)
            print(len(gold), gold)
            print(len(system), system)
            raise ValueError("The given files to not contain the same number of words on line '{}'".format(" ".join(text)))

        # Accuracy
        acc_total += len(gold)
        for i in range(len(gold)):
            if gold[i] == system[i]:
                acc_correct += 1
            else:
                if verbose:
                    print(gold, system)
                    print(gold[i], system[i])

        # F-score
        for i in range(len(gold)):
            if gold[i] != text[i]:
                f_gold += 1
                if gold[i] == system[i]:
                    f_both += 1
            if system[i] != text[i]:
                f_system += 1

    if f_system == 0:
        f_system = 1e-6
    accuracy = 100 * acc_correct / acc_total
    precision = 100 * f_both / f_system
    recall = 100 * f_both / f_gold
    f1_score = 100 * 2 * f_both / (f_system + f_gold)
    total_words = acc_total
    gold_corrections = f_gold
    # Print results
    # print("Words: {}, accuracy: {:.2f}%".format(acc_total, 100 * acc_correct / acc_total))
    # print("Gold corrections: {}, precision: {:.2f}%, recall: {:.2f}%, f-score: {:.2f}%".format(f_gold, 100 * f_both / f_system, 100 * f_both / f_gold,
    #                                                                                            100 * 2 * f_both / (f_system + f_gold)))
    return accuracy, precision, recall, f1_score, total_words, gold_corrections


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("original_file", type=argparse.FileType("r"),
                        help="Name of the file with the original text data.")
    parser.add_argument("gold_file", type=argparse.FileType("r"),
                        help="Name of the file with the gold data.")
    parser.add_argument("system_file", type=argparse.FileType("r"), nargs="?", default=sys.stdin,
                        help="Name of the file with the gold data.")
    args = parser.parse_args()

    original_file, gold_file, system_file = args.original_file, args.gold_file, args.system_file
    accuracy, precision, recall, f1_score, total_words, gold_corrections = evaluate(original_file, gold_file, system_file)

    print("Words: {}, accuracy: {:.2f}%".format(total_words, accuracy))
    print("Gold corrections: {}, precision: {:.2f}%, recall: {:.2f}%, f-score: {:.2f}%".format(gold_corrections, precision, recall, f1_score))
