#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import argparse
import sys

import diacritization_stripping_data

parser = argparse.ArgumentParser()
parser.add_argument("--uninorms", action="store_true",
                    help="Use diacritization stripping based on Unicode Normalization")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Also compute statistics of the stripping")
args = parser.parse_args()

maps = []
if not args.uninorms:
    maps.append(diacritization_stripping_data.strip_diacritization_uninames)
else:
    maps.append(diacritization_stripping_data.strip_diacritization_uninorms)

total, stripped, stripped_map = 0, 0, {}
for line in sys.stdin:
    output = ""
    for c in line:
        for m in maps:
            if c in m:
                stripped += 1
                stripped_map[c] = stripped_map.get(c, 0) + 1
                output += m[c]
                break
        else:
            output += c

        if not c.isspace():
            total += 1
    print(output, end="")

if args.verbose:
    histogram = sorted(stripped_map.items(), key=lambda key_value: key_value[1], reverse=True)[:10]
    histogram = " ".join(map(lambda key_value: "{}:{:.2f}%".format(key_value[0], 100 * key_value[1] / stripped), histogram))

    print("Total: {}, stripped: {:.2f}%, histogram: {}".format(total, 100 * stripped / total, histogram),
          file=sys.stderr)
