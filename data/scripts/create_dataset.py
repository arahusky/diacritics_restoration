import argparse
import numpy as np
import os
import io
import sys

'''
input stream:
    - number of lines
    - lines

for train: at least given number of characters, enough of diacritics
split into train/dev/test
'''


def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("num_lines", type=int, help="Number of lines of incoming stream.")
    parser.add_argument("output_dir", type=str, default="dataset", help="Path to folder to store created dataset.")
    parser.add_argument("--min_chars", type=int, default=0,
                        help="Minimal number of characters the training sentence has to have.")
    parser.add_argument("--dev_sentences", type=int, default=20000, help="Number of sentences in dev set.")
    parser.add_argument("--test_sentences", type=int, default=15000, help="Number of sentences in test set.")
    parser.add_argument("--filename_train", type=str, default='target_train.txt', help="")
    parser.add_argument("--filename_dev", type=str, default='target_dev.txt', help="")
    parser.add_argument("--filename_test", type=str, default='target_test.txt', help="")
    args = parser.parse_args()

    min_chars = args.min_chars
    num_dev_sentences = args.dev_sentences
    num_test_sentences = args.test_sentences

    # create directory for saving dataset
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_target_file = os.path.join(output_dir, args.filename_train)
    dev_target_file = os.path.join(output_dir, args.filename_dev)
    test_target_file = os.path.join(output_dir, args.filename_test)

    line_count = args.num_lines
    print('Processing: {} sentences'.format(line_count))
    # line_permutation = np.random.permutation(line_count) -- permutation does not make sense here as testing sentences should be from non-seen articles
    line_permutation = range(line_count)
    dev_indices = line_permutation[:num_dev_sentences]
    test_indices = line_permutation[num_dev_sentences:num_dev_sentences + num_test_sentences]
    train_indices = line_permutation[num_dev_sentences + num_test_sentences:]

    line_index = 0
    with io.open(train_target_file, 'w', encoding='utf8') as train_target_writer, \
            io.open(dev_target_file, 'w', encoding='utf8') as dev_target_writer, \
            io.open(test_target_file, 'w', encoding='utf8') as test_target_writer:

        for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8'):
            # remove leading and trailing whitespaces
            line = line.strip()

            # convert to lower-case
            line = line.lower()

            current_target_writer = None
            if line_index in train_indices:
                current_target_writer = train_target_writer

                if len(line) < min_chars:
                    line_index += 1
                    continue

            elif line_index in dev_indices:
                current_target_writer = dev_target_writer
            elif line_index in test_indices:
                current_target_writer = test_target_writer
            else:
                # should never happen
                raise ValueError('Something went terribly wrong.')

            current_target_writer.write(u'{}\n'.format(line))

            line_index += 1

            # status
            if line_index % (line_count // 100) == 0:
                print('Progress: {}%'.format(line_index // (line_count // 100)))


if __name__ == "__main__":
    main()
