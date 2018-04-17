import argparse
import numpy as np
import os
import io
import sys
import hashlib


def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("first_file", type=str, help="Bigger file -- will be reduced")
    parser.add_argument("second_file", type=str, help="Smaller file.")
    args = parser.parse_args()

    second_file_line_hashes = set()
    with io.open(args.second_file, 'r', encoding='utf8') as reader:
        for line in reader:
            line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
            second_file_line_hashes.add(line_hash)

    with io.open(args.first_file, 'r', encoding='utf8') as reader:
        with io.open(args.first_file + '_temp', 'w', encoding='utf8') as writer:
            for line in reader:
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                if not line_hash in second_file_line_hashes:
                    writer.write(line)


if __name__ == "__main__":
    main()
