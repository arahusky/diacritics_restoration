#!python-master-170829/bin/python3.7
from __future__ import division
from __future__ import print_function

import unicodedata

stripping_data = open("diacritization_stripping_data.py", mode="w", encoding="utf-8")

print("# coding=utf-8\n", file=stripping_data)

print("strip_diacritization_unidata_version = {}\n".format(repr(unicodedata.unidata_version)), file=stripping_data)

print("strip_diacritization_uninorms = {", file=stripping_data)
for ord in range(1, 2**20 + 2**16):
    original = chr(ord)
    decomposed = unicodedata.normalize("NFD", original)
    if original == decomposed:
        continue

    decomposed_stripped = ""
    for c in decomposed:
        if not unicodedata.category(c).startswith("M"):
            decomposed_stripped += c
    stripped = unicodedata.normalize("NFC", decomposed_stripped)

    if original != stripped:
        print("  {}: {},".format(repr(original), repr(stripped)), file=stripping_data)
print("}\n", file=stripping_data)

print("strip_diacritization_uninames = {", file=stripping_data)
for ord in range(1, 2**20 + 2**16):
    original = chr(ord)
    name = unicodedata.name(original, None)
    if name is None:
        continue

    name_with_index = name.find(" WITH ")
    if name_with_index < 0:
        continue
    name_stripped = name[0:name_with_index]

    try:
        stripped = unicodedata.lookup(name_stripped)
    except KeyError:
        continue

    print("  {}: {},".format(repr(original), repr(stripped)), file=stripping_data)
print("}\n", file=stripping_data)
